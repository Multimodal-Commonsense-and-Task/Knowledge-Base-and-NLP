"""
LangChain ChatModel classes for models provided via OpenRouter API.
Uses OpenAI-compatible API, inheriting from ChatOpenAI with fixed base_url and model.
"""

from langchain_openai import ChatOpenAI


class Llama3_70B(ChatOpenAI):
    """OpenRouter: meta-llama/llama-3.3-70b-instruct"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.setdefault("model", "meta-llama/llama-3.3-70b-instruct")
        # API key is injected from model_factory
        super().__init__(**kwargs)


class Qwen3_32B(ChatOpenAI):
    """OpenRouter: qwen/qwen3-32b"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.setdefault("model", "qwen/qwen3-32b")
        # API key is injected from model_factory
        super().__init__(**kwargs)


class Ministral_14B(ChatOpenAI):
    """OpenRouter: mistralai/ministral-14b-2512"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.setdefault("model", "mistralai/ministral-14b-2512")
        # API key is injected from model_factory
        super().__init__(**kwargs)

class Gemini_3_Pro(ChatOpenAI):
    """OpenRouter: google/gemini-3-pro"""

    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        kwargs.setdefault("model", "google/gemini-3-pro-preview")
        # API key is injected from model_factory
        super().__init__(**kwargs)