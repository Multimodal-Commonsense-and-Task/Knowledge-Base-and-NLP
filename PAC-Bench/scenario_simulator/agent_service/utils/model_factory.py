import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from utils.vllm_models import Ministal3_14B, Llama3_70B, Qwen3_32B, Qwen3_8B

load_dotenv()

PROVIDER_CONFIG = {
    "gpt": {
        "model_provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude": {
        "model_provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    # "llama3-70b": {
    #     "model_provider": "openai",
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "model_id": "meta-llama/llama-3.3-70b-instruct",
    #     "api_key_env": "OPENROUTER_API_KEY",
    # },
    # "qwen3-32b": {
    #     "model_provider": "openai",
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "model_id": "qwen/qwen3-32b",
    #     "api_key_env": "OPENROUTER_API_KEY",
    # },
    # "ministral-14b": {
    #     "model_provider": "openai",
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "model_id": "mistralai/ministral-14b-2512",
    #     "api_key_env": "OPENROUTER_API_KEY",
    # },
    "llama3-70b": {
        "model_provider": "openai",
        "vllm": Llama3_70B,
    },
    "ministral-14b": {
        "model_provider": "openai",
        "vllm": Ministal3_14B,
    },
    "qwen3-32b": {
        "model_provider": "openai",
        "vllm": Qwen3_32B,
    },
    "qwen3-8b": {
        "model_provider": "openai",
        "vllm": Qwen3_8B,
    },
}

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, **kwargs):
        for provider_key, config in PROVIDER_CONFIG.items():
            if provider_key in model_name:
                model_provider = config["model_provider"]
                api_key_env = config.get("api_key_env")
                base_url = config.get("base_url")
                actual_model_name = config.get("model_id", model_name)

                model_kwargs = dict(kwargs)

                if "vllm" in config:
                    vllm_class = config["vllm"]
                    return vllm_class(**model_kwargs)

                if api_key_env and "api_key" not in model_kwargs:
                    env_api_key = os.getenv(api_key_env)
                    if env_api_key:
                        model_kwargs["api_key"] = env_api_key

                if "api_key" not in model_kwargs:
                    raise ValueError(
                        f"Missing API key for provider '{provider_key}'. Set '{api_key_env}'."
                    )

                if base_url and "base_url" not in model_kwargs:
                    model_kwargs["base_url"] = base_url
                    if "openrouter.ai" in base_url:
                        extra_body = model_kwargs.get("extra_body", {})
                        extra_body["reasoning"] = {"exclude": True}
                        model_kwargs["extra_body"] = extra_body

                return init_chat_model(
                    actual_model_name,
                    model_provider=model_provider,
                    **model_kwargs
                )
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    # Example usage
    model = ModelFactory.create_model("claude-sonnet-4.5")
    print(model)