from utils import *
from openai import OpenAI
import httpx  # 添加这行导入

class Percipient:
    def __init__(
        self,
        openai_key,
        memory,
        question_model_name="gpt-4o",
        answer_method="active",     # active or caption
        answer_model="gpt-4o",        # mllm or gpt-4o
        answer_mllm_url=None,       # mllm url
        temperature=0
    ):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.question_model_name = question_model_name
        self.answer_model = answer_model

        self.temperature = temperature
        self.memory = memory
        self.answer_method = answer_method
        self.temperature = temperature

        assert self.answer_method in ["active", "caption"], "Please input correct method of perception"
        assert self.answer_model in ["mllm", "gpt-vision", "gpt-4o"], "Please input correct model of perception"
        assert self.memory is not None, "Please input memory"

        if self.answer_model == "mllm":
            assert answer_mllm_url is not None, "Please input mllm url"
            self.mllm = MineLLM(answer_mllm_url=answer_mllm_url)

        elif self.answer_model == "gpt-4o":
            assert answer_model is not None, "Please input gpt-4o model name"
            # 设置HTTP代理
            self.client = OpenAI(
                api_key=openai_key,
                http_client=httpx.Client(
                    proxy="http://127.0.0.1:7891"
                )
            )

        else:
            raise ValueError("Percipient's answer mllm is incorrect.")

    def perception_llm(self, messages):
        print("now is perception_llm!!!")  # 
        # import pdb; pdb.set_trace()
        response = self.client.chat.completions.create(
            model=self.question_model_name,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}  # 添加这一行

        )
        res = response.choices[0].message
        return res

    def perception_vlm(self, vlm_question, image_path):
        # 是单图的感知策略 
        if self.answer_model != "gpt-4o":
            raise ValueError("This method is only for GPT4-O model")
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("now is perception_vlm!!!")
        # import pdb; pdb.set_trace()
        gpt_vision_system_message = load_prompt("gpt-vision-active_perception_system")
        response = self.client.chat.completions.create(
            model=self.answer_model,
            messages=[
                {"role": "system", "content": gpt_vision_system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": vlm_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        res = response.choices[0].message.content
        return res

    def perceive(self, task_information, find_obj, file_path, max_retries=5):
        if self.answer_method == "active":
            return self.activate_perception(task_information=task_information, find_obj=find_obj, file_path=file_path, max_retries=max_retries)
        elif self.answer_method == "caption":
            return self.caption_perception(task_information=task_information, find_obj=find_obj, file_path=file_path, max_retries=max_retries)
        else:
            raise ValueError("Percipient's answer method is incorrect.")


    def get_perception_question(self, task_information, find_obj):
        # Active Perception
        active_perception_system = load_prompt("active_perception_system")
        active_perception_query = load_prompt("active_perception_query").format(
            task_information=task_to_description_prompt(update_find_task_prompt(task_information, find_obj)), 
            current_environment_information=list_dict_to_prompt(self.memory.current_environment_information)
        )

        # messages = [
        #     SystemMessage(content=active_perception_system),
        #     HumanMessage(content=active_perception_query)
        # ]   
        messages = {
            "system": active_perception_system,
            "user": active_perception_query
        }

        question_info = self.perception_llm(messages).content
        return fix_and_parse_json(question_info)



    def check_perception_question(self, question_dict):
        question_dict["status"] = int(question_dict["status"])
        ## Check if the Active Perception is over
        assert question_dict["status"] in [0, 1, 2]


        ## Success
        if question_dict["status"] == 2:
            return 2

        ## Failure
        if question_dict["status"] == 0:
            return 0

        ## Asked before
        for information in self.memory.current_environment_information:
            question_type = question_dict["query"]["type"]
            if question_type in information["type"] or information["type"] in question_type:
                return 3

        # Continue asking
        return 1
    

    def activate_perception(self, task_information, find_obj, file_path, max_retries=5):

        # Active Perception Failure
        if max_retries == 0:
            log_info("************Failed to get activate perception. Consider updating your prompt.************\n\n")
            return 

        try:
            ## Get the environment information
            while True:
                ## Current Environment State
                log_info(f"Current Environment Information: \n{list_dict_to_prompt(self.memory.current_environment_information)}")

                ## Active Perception
                # 获取感知问题，因为感知的画面中有太多的物体，不能都详细的描述的出来，而是要关注特定的object
                question_dict = self.get_perception_question(task_information, find_obj)

                ## Check if the Active Perception is over
                check_result = self.check_perception_question(question_dict)
                if check_result == 2:
                    log_info(f"Active Perception Success: {question_dict}")
                    log_info("************Active Perception Finish!************\n")
                    return 2
                elif check_result == 3:
                    log_info(f"************Active Perception Failure: The question about {question_dict['query']['type']} was asked before. Continue finding************\n")
                    return 0
                elif check_result == 0:
                    log_info(f"************Active Perception Failure: {question_dict['thoughts']}***********\n")
                    # import pdb; pdb.set_trace()
                    return 0     
                        
                ## Active Perception Question
                print("\033[34m" + f"Active Perception Question: {question_dict['query']['question']}" + "\033[0m")
                log_info(f"GPT Question: {question_dict['query']['question']}")

                ## Interact with gpt-4o
                # 根据question_dict['query']['question']，获取感知结果
                answer = self.perception_vlm(question_dict['query']['question'], file_path)

                log_info(f"gpt-4o Answer: {answer}")

                ## record the information
                self.memory.current_environment_information.append({
                        "type": question_dict['query']['type'],
                        "info": answer
                    })

        except Exception as e:
            log_info(f"Error arises in Active Perception part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.activate_perception(
                task_information=task_information,
                find_obj=find_obj,
                file_path=file_path,
                max_retries=max_retries - 1,
            )

    # 下面是caption感知，暂无使用

    def check_caption_perception(self, task_information, find_obj):
        # Caption Perception
        check_caption_perception_system = load_prompt("check_caption_perception_system")
        check_caption_perception_query = load_prompt("check_caption_perception_query").format(
            task_information=task_to_description_prompt(update_find_task_prompt(task_information, find_obj)), 
            current_environment_information=list_dict_to_prompt(self.memory.current_environment_information)
        )

        # messages = [
        #     SystemMessage(content=check_caption_perception_system),
        #     HumanMessage(content=check_caption_perception_query)
        # ]   
        messages = {
            "system": check_caption_perception_system,
            "user": check_caption_perception_query
        }

        question_info = self.perception_llm(messages).content
        return fix_and_parse_json(question_info)

    def caption_perception(self, task_information, find_obj, file_path, max_retries=5):

        # Caption Perception Failure
        if max_retries == 0:
            log_info("************Failed to get activate perception. Consider updating your prompt.************\n\n")
            return

        try:
            ## Interact with MLLM
            answer = self.mllm.query("Could you describe this Minecraft image?", file_path)

            log_info(f"MLLM Answer: {answer}")

            ## record the information
            self.memory.current_environment_information.append({
                    "type": "environment caption",
                    "info": answer
                })
            
            ## Check if finish perception
            check_dict = self.check_caption_perception(task_information, find_obj)
            if check_dict["status"] == 2:
                log_info(f"Active Perception Success: {check_dict}")
                log_info("************Active Perception Finish!************\n")
                return 2
            else:
                log_info(f"************Active Perception Failure: {check_dict['thoughts']}***********\n")
                return 0 

        except Exception as e:
            log_info(f"Error arises in Caption Perception part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.caption_perception(
                task_information=task_information,
                find_obj=find_obj,
                file_path=file_path,
                max_retries=max_retries - 1,
            )