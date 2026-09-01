import os
import re
import time
from typing import List, Dict

import openai
#from dotenv import load_dotenv
#from openai import OpenAI
from tenacity import wait_exponential, retry, stop_after_attempt, RetryCallState

from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
    wait_exponential_jitter,
    retry_if_exception_type,
) 

SLEEP_TIME=10
TEMPERATURE=0.2

def re_extract(prefix: str, target: str, truncate: bool = False):
    reg = rf"[\'\"]?{prefix}[\'\"]?: ([\w\W]+)"

    next_line_removed_target = target.split("\n")[0].strip() if truncate else target.strip()
    res = re.findall(reg, next_line_removed_target)
    return_target = res[0] if len(res) >= 1 else next_line_removed_target
    return_target = return_target.replace("\t", " ")

    if (return_target.startswith('"') and return_target.endswith('"')) \
            or (return_target.startswith("'") and return_target.endswith("'")):
        return_target = return_target[1:-1]

    return return_target


def clean_query(s: str):
    if s.startswith("'") or s.startswith('"'):
        s = s[1:]
    if s.endswith("'") or s.endswith('"'):
        s = s[:-1]
    return s


def clean_queries(ss: List[str]):
    return [clean_query(s) for s in ss]


def clean_response(s: str):
    # reg = rf"\[+([\w\W]+)\]+"
    if "Expand[Query]: " in s:
        s = s.split("Expand[Query]: ")[-1]
    if "[" in s and "]" in s:
        s = s.split("[")[-1].split("]")[0]
        clean_res = clean_queries(s.split(", "))
    elif "[" in s:
        s = s.split("[")[-1]
        clean_res = clean_queries(s.split(", "))
    elif "]" in s:
        s = s.split("]")[0]
        clean_res = clean_queries(s.split(", "))
    else:
        clean_res = s
    return clean_res


def check_before(retry_state: RetryCallState):
    if retry_state.attempt_number == 1:
        retry_state.args[0].request_timeout = 3
        
    # _client = retry_state.args[0].clients[retry_state.args[0].current]
    # api_key = _client.api_key
    # if openai.api_key == retry_state.args[0].api_key_openai and retry_state.attempt_number <= 3:
    #     print("Restoring API key... to Azure")
    #     openai.api_key = retry_state.args[0].api_key
    #     openai.api_type = retry_state.args[0].api_type
    #     openai.api_base = retry_state.args[0].api_base
    #     openai.api_version = retry_state.args[0].api_version


def log_attempt_number(retry_state: RetryCallState):
    """return the result of the last call attempt"""
    if type(retry_state.outcome.exception()) is openai.APITimeoutError:
        print("Increasing timeout...")
        retry_state.args[0].request_timeout += 1
        return

    if 1 < retry_state.attempt_number <= 10:
        if retry_state.args[0].current == "azure1":
            print("Switching API key... => azure2")
            retry_state.args[0].current = "azure2"
        else:
            print("Switching API key... => azure1")
            retry_state.args[0].current = "azure1"

    # elif retry_state.attempt_number > 3:
    #     print("Switching API key... => openai")
    #     retry_state.args[0].current = "openai"

    print(f"Retrying: {retry_state.attempt_number}...")


class GPT:
    # text-davinci-003 is also good but expensive.
    # gpt-3.5-turbo is set as default.
    def __init__(self,
                 max_token_length: int = 200,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 stop: List[str] = None,
                 max_iter: int = 1
                 ):
        self.stop = stop
        if self.stop is None:
            self.stop = ["###"]

        self.max_token_length = max_token_length
        self.temperature = temperature
        self.top_p = top_p

        self.max_iter = max_iter

        #load_dotenv() # load .env file

        self.request_timeout = 3

        #self.current = "azure1"
        #self.current = "openai"

        self.request_timeout = 3

        

        self.model_name = {
            "azure1": os.getenv("AZURE_OPENAI_MODEL_NAME"),
            "azure2": os.getenv("AZURE_OPENAI_MODEL_NAME"),
            "openai": os.getenv("OPENAI_API_MODEL_NAME")
        }

    """
    @retry(wait=wait_exponential(multiplier=10, min=5, max=10), stop=stop_after_attempt(7),
           before=check_before,
           after=log_attempt_number)    
    def query(self, prompt: List[Dict[str, str]]):
        #current_client: self.clients[self.current]

        response = current_client.chat.completions.create(
            model=self.model_name[self.current],
            messages=prompt,
            max_tokens=self.max_token_length,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )
        return response.choices[0].message.content
    """

    def set_api(self, api_name: str, model_name: str):
        self.current_api = api_name
        self.curr_model_name = model_name

        if api_name == "openai":
            print(f"Using OpenAI API, model {model_name}...")
            openai.api_key = os.getenv("OPENAI_API_KEY") # mnskim0
            openai.api_type = 'open_ai'
            openai.api_base = 'https://api.openai.com/v1'
            openai.api_version = None
        elif api_name == "openai_codex":    
            print(f"Using OpenAI Codex API, model {model_name}...")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.api_type = 'open_ai'
            openai.api_base = 'https://api.openai.com/v1'
            openai.api_version = None
        elif api_name == "azure1":
            print(f"Using Azure API, model {model_name}...")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY_1")
            openai.api_type = "azure"
            openai.api_base = "https://ldi2023auginstance1.openai.azure.com/"
            openai.api_version = "2023-05-15"
        elif api_name == "azure2":
            print(f"Using Azure API, model {model_name}...")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY_2")
            openai.api_type = "azure"
            openai.api_base = ""
            openai.api_version = "2023-05-15"
        
    @retry(wait=wait_exponential(multiplier=10, min=5, max=10), stop=stop_after_attempt(7),
           before=check_before,
           after=log_attempt_number) 
    def query(self, prompt):
        #openai.api_key = os.getenv("OPENAI_API_KEY") # mnskim0
        #openai.api_key = os.getenv("OPENAI_API_KEY") # mnskim0

        #openai.api_type = 'open_ai'
        #openai.api_base = 'https://api.openai.com/v1'
        #openai.api_version = None
        #model = "gpt-3.5-turbo-0301"
        #model = "gpt-4-0613"
        time.sleep(SLEEP_TIME)
        if self.current_api == "openai":
            response = openai.ChatCompletion.create(
            model=self.curr_model_name,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_token_length,
            top_p=self.top_p,
            stop=self.stop,                                                
            )
            return response.choices[0].message.content
        elif self.current_api == "azure1":
            response = openai.ChatCompletion.create(
                engine=self.curr_model_name,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_token_length,
                top_p=self.top_p,
                stop=self.stop,                                                
            )
            return response.choices[0].message.content
        

if __name__ == '__main__':
    a = "Expand[Query]: ['online breast cancer community', 'longitudinal analysis', 'discussion topics', 'convolutional neural networks']"
    b = re_extract("Expand\[Query\]", a)
    c = clean_response(b)
    d = f'[{", ".join(c)}]'
    print()

    gpt = GPT()

    b = gpt.query([
        {
            "role": "user",
            "content": a,
        }
    ])
    print(b)

