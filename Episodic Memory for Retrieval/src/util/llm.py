import tiktoken

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputParam
from transformers import AutoTokenizer
from typing import cast, List

from src.util.const import OPENAI_MODELS, VLLM_MODELS

class LLM:
    def __init__(self, model_name : str, port : int = 8000):
        self.model_name = model_name
        if model_name in OPENAI_MODELS:
            self.llm = OpenAILLM(model_name)
        elif model_name in VLLM_MODELS:
            self.llm = vLLM(model_name, port)
        else:
            raise ValueError(f'Unsupported model name: [{model_name}]')

    def generate(self, messages: list[dict[str, str]], max_input_tokens : int = 36000, remove_space: bool = False, **kwargs):
        return self.llm.generate(messages=messages, max_input_tokens=max_input_tokens, remove_space=remove_space, **kwargs)

class OpenAILLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = self.load_api_key('.env/openai_api_key.txt')
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def load_api_key(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as fp:
            api_key = fp.read().strip()
        return api_key

    def generate(self, messages: list[dict[str, str]], max_input_tokens : int = 36000, remove_space: bool = False, **kwargs) -> tuple[str, int, int]:
        for i in range(len(messages)):
            if messages[i]['role'] == 'system':
                messages[i]['role'] = 'developer'
        num_input_tokens = sum(len(self.tokenizer.encode(msg['content'])) for msg in messages)
        input_params = cast(ResponseInputParam, messages)
        response = self.client.responses.create(model=self.model_name, input=input_params, **kwargs)
        output = response.output_text.strip()
        num_output_tokens = len(self.tokenizer.encode(output))
        if remove_space:
            output = ' '.join(output.split())
        return output, num_input_tokens, num_output_tokens

class vLLM:
    MODEL_MAPPING = {
        'qwen3': 'Qwen/Qwen3-32B-FP8',
        'qwen2.5': 'Qwen/Qwen2.5-32B-Instruct'
    }

    def __init__(self, model_name : str, port : int = 8000):
        if model_name not in self.MODEL_MAPPING:
            raise ValueError(f'Unsupported vLLM model name: [{model_name}]')
        self.model_name = vLLM.MODEL_MAPPING[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.client = OpenAI(api_key='EMPTY', base_url=f'http://localhost:{port}/v1')

    def generate(self, messages: list[dict[str, str]], max_input_tokens : int = 36000, remove_space: bool = False, **kwargs) -> tuple[str, int, int]:
        messages.append({'role': 'assistant', 'content': '<think>Okay, I think I have finished thinking.</think> {\n'})
        num_input_tokens = sum(len(self.tokenizer.encode(msg['content'])) for msg in messages)
        while num_input_tokens > max_input_tokens:
            messages[0]['content'] = messages[0]['content'][:-100] + '...'
            num_input_tokens = sum(len(self.tokenizer.encode(msg['content'])) for msg in messages)
        if not 'extra_body' in kwargs:
            kwargs['extra_body'] = {'top_k': 20, 'max_tokens': 4096}
        input_params = cast(List[ChatCompletionMessageParam], messages)
        response = self.client.chat.completions.create(model=self.model_name, messages=input_params, **kwargs)
        output = response.choices[0].message.content.strip()
        num_output_tokens = len(self.tokenizer.encode(output))
        if '</think>' in output:
            output = output.split('</think>')[-1].strip()
        if remove_space:
            output = ' '.join(output.split())
        return output, num_input_tokens, num_output_tokens
