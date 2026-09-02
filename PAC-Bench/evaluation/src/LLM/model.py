import os
from dotenv import load_dotenv
from openai import OpenAI

from typing import Any, Dict, Optional

load_dotenv()

class LLM:
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.name = model_name or "gpt-5.2"
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content

        usage = getattr(response, "usage", None)
        prompt_tokens: Optional[int] = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens: Optional[int] = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens: Optional[int] = getattr(usage, "total_tokens", None) if usage else None

        token_usage: Dict[str, Optional[int]] = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": total_tokens,
        }

        return {
            "content": content,
            "token_usage": token_usage,
        }