"""
LangChain ChatModel classes for models deployed on vLLM server.
Uses OpenAI-compatible API, inheriting from ChatOpenAI with fixed base_url and model.
"""

from langchain_openai import ChatOpenAI


class Llama3_8B(ChatOpenAI):
    """vLLM: unsloth/Meta-Llama-3.1-8B-Instruct"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "http://example/v1")
        kwargs.setdefault("model", "unsloth/Meta-Llama-3.1-8B-Instruct")
        kwargs.setdefault("api_key", "EMPTY")
        super().__init__(**kwargs)


class Llama3_70B(ChatOpenAI):
    """vLLM: Llama-3.3-70B-Instruct"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "http://example/v1")
        kwargs.setdefault("model", "Llama-3.3-70B-Instruct")
        kwargs.setdefault("api_key", "EMPTY")
        super().__init__(**kwargs)


class Qwen3_8B(ChatOpenAI):
    """vLLM: unsloth/Qwen3-8B"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "http://example/v1")
        kwargs.setdefault("model", "unsloth/Qwen3-8B")
        kwargs.setdefault("api_key", "EMPTY")
        kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        super().__init__(**kwargs)


class Qwen3_32B(ChatOpenAI):
    """vLLM: Qwen/Qwen3-32B"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "http://example/v1")
        kwargs.setdefault("model", "Qwen/Qwen3-32B")
        kwargs.setdefault("api_key", "EMPTY")
        kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        super().__init__(**kwargs)

class Ministal3_14B(ChatOpenAI):
    """vLLM: ministral-3-14b-instruct"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "http://example/v1")
        kwargs.setdefault("model", "ministral-3-14b-instruct")
        kwargs.setdefault("api_key", "EMPTY")
        super().__init__(**kwargs)
