"""
LLM model abstraction and service-specific implementations
Supports OpenAI, OpenRouter, vLLM, Claude, and Gemini
"""
from abc import ABC, abstractmethod
import time
import base64
import io
import re
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path
import logging
import os

# Disable langchain verbose logging
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_LOG"] = "false"

# Suppress langchain logging
logging.getLogger("langchain").setLevel(logging.CRITICAL)
logging.getLogger("langchain_openai").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Suppress any print statements from langchain internals
import sys
from io import StringIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
    PIL_Image = Image.Image
except ImportError:
    PIL_AVAILABLE = False
    PIL_Image = None

# Import libraries for each service
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False


MAX_RETRY = 3
RETRY_SLEEP = 5


def image_to_base64_url(image: Union[str, Image.Image, bytes]) -> str:
    """
    Convert image to base64 data URL
    
    Args:
        image: Image path (str), PIL Image, or bytes
        
    Returns:
        Base64 data URL string
    """
    if isinstance(image, str):
        # Assume it's a file path
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, Image.Image):
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for PIL Image processing. pip install pillow")
        # Convert PIL Image to bytes
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def process_multimodal_content(content: Union[str, List[Dict[str, Any]]], images: Optional[List[Union[str, Image.Image, bytes]]] = None) -> List[Dict[str, Any]]:
    """
    Process content with optional images for multimodal messages
    
    Args:
        content: Text content (str) or already formatted content list
        images: List of images (paths, PIL Images, or bytes)
        
    Returns:
        Formatted content list for multimodal messages
    """
    if isinstance(content, list):
        # Already formatted
        return content
    
    if images is None or len(images) == 0:
        # No images, return simple text
        return [{"type": "text", "text": content}]
    
    # Convert images to base64 URLs
    image_urls = [image_to_base64_url(img) for img in images]
    
    # Create multimodal content
    multimodal_content = []
    
    # Check if content has image placeholder
    if "<IMAGE_PLACEHOLDER>" in content:
        parts = content.split("<IMAGE_PLACEHOLDER>")
        for i, part in enumerate(parts):
            if part.strip():
                multimodal_content.append({"type": "text", "text": part})
            # Add images after each text part (except the last one)
            if i < len(image_urls):
                multimodal_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_urls[i], "detail": "low"}
                })
        # Add remaining images if any
        for img_url in image_urls[len(parts)-1:]:
            multimodal_content.append({
                "type": "image_url",
                "image_url": {"url": img_url, "detail": "low"}
            })
    else:
        # Add text first, then all images
        multimodal_content.append({"type": "text", "text": content})
        for img_url in image_urls:
            multimodal_content.append({
                "type": "image_url",
                "image_url": {"url": img_url, "detail": "low"}
            })
    
    return multimodal_content


class BaseLLMModel(ABC):
    """Base abstract class for all LLM models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary
                - model_name: Model name
                - temperature: Temperature setting
                - max_tokens: Maximum number of tokens
                - api_key: API key (if required)
                - base_url: Base URL (if required)
        """
        self.config = config
        self.model_name = config.get("model_name", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 8192)
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Initialize the model"""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate response from messages
        
        Args:
            messages: List of messages. Each message can be:
                - {"role": "user", "content": "text"} for text only
                - {"role": "user", "content": "text", "images": [image1, image2]} for multimodal
                - {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]} for pre-formatted
            **kwargs: Additional parameters
            
        Returns:
            {
                "content": str,  # Generated text
                "usage": dict,   # Token usage information
                "cost": float    # Cost in USD
            }
        """
        pass
    
    def generate_with_retry(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate with retry logic"""
        for attempt in range(MAX_RETRY):
            try:
                return self.generate(messages, **kwargs)
            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    print(f"Error on attempt {attempt + 1}/{MAX_RETRY}: {e}. Retrying...")
                    time.sleep(RETRY_SLEEP)
                else:
                    print(f"Failed after {MAX_RETRY} attempts: {e}")
                    raise
    
    def _process_messages_for_multimodal(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process messages to handle multimodal content (images) - default implementation"""
        processed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # If content is already a list (multimodal format), use it as is
            if isinstance(content, list):
                processed.append({"role": role, "content": content})
            elif isinstance(content, str):
                # Check if message has images
                images = msg.get("images", [])
                if images:
                    processed_content = process_multimodal_content(content, images)
                    processed.append({"role": role, "content": processed_content})
                else:
                    processed.append({"role": role, "content": content})
            else:
                processed.append(msg)
        return processed


class OpenAIModel(BaseLLMModel):
    """OpenAI API model"""
    
    def _setup(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is not installed. pip install openai")
        
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Process messages to handle multimodal content
        processed_messages = self._process_messages_for_multimodal(messages)
        
        # Some newer OpenAI models (e.g., gpt-5.1) use max_completion_tokens instead of max_tokens
        # Check if model name suggests it needs max_completion_tokens
        use_max_completion_tokens = (
            "gpt-5" in self.model_name.lower() or
            "o3" in self.model_name.lower() or
            kwargs.get("use_max_completion_tokens", False)
        )
        
        # Prepare parameters
        create_params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature,
        }
        
        if use_max_completion_tokens:
            create_params["max_completion_tokens"] = max_tokens
        else:
            create_params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**create_params)
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        # Calculate cost (simple example, may vary by model)
        cost = self._calculate_cost(usage)
        
        return {
            "content": content,
            "usage": usage,
            "cost": cost,
        }
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage (may vary by model)"""
        # Default values, should reference model-specific pricing in practice
        input_cost_per_1k = 0.01
        output_cost_per_1k = 0.03
        return (usage["prompt_tokens"] * input_cost_per_1k / 1000 + 
                usage["completion_tokens"] * output_cost_per_1k / 1000)


class OpenRouterModel(BaseLLMModel):
    """OpenRouter API model (OpenAI compatible)"""
    
    def _setup(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is not installed. pip install openai")
        
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        
        base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Process messages to handle multimodal content
        processed_messages = self._process_messages_for_multimodal(messages)
        
        # OpenRouter can include additional information in HTTP headers
        headers = {
            "HTTP-Referer": self.config.get("http_referer", ""),
            "X-Title": self.config.get("app_name", "EMBGuard"),
        }
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=headers,
        )
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        cost = self._calculate_cost(usage)
        
        return {
            "content": content,
            "usage": usage,
            "cost": cost,
        }
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate OpenRouter cost (may vary by model)"""
        # OpenRouter pricing varies by model, should fetch pricing from API response in practice
        input_cost_per_1k = 0.01
        output_cost_per_1k = 0.03
        return (usage["prompt_tokens"] * input_cost_per_1k / 1000 + 
                usage["completion_tokens"] * output_cost_per_1k / 1000)


class VLLMModel(BaseLLMModel):
    """vLLM model using langchain_openai ChatOpenAI"""
    
    def _setup(self):
        if not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError("langchain_openai library is not installed. pip install langchain-openai")
        
        base_url = self.config.get("base_url")
        if not base_url:
            raise ValueError("base_url is required for vLLM model")
        
        api_key = self.config.get("api_key", "EMPTY")  # vLLM may not require API key, but ChatOpenAI needs one
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=self.temperature,
            timeout=300,
            verbose=False
        )
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Use kwargs if provided, otherwise use instance defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Temporarily suppress all stdout/stderr to prevent image data from being printed
        from contextlib import redirect_stdout, redirect_stderr
        
        # Process messages to handle multimodal content (suppress output here too)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            processed_messages = self._process_messages_for_multimodal(messages)
        
        # If parameters differ from instance defaults, create a new instance
        if temperature != self.temperature or max_tokens != self.max_tokens:
            base_url = self.config.get("base_url")
            api_key = self.config.get("api_key", "EMPTY")
            
            llm = ChatOpenAI(
                model=self.model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=300,
                verbose=False
            )
        else:
            llm = self.llm
        
        # Suppress output during invoke to prevent image data from being printed
        # Also suppress during message processing
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            response = llm.invoke(processed_messages)
        
        content = response.content
        usage = {
            "prompt_tokens": response.response_metadata["token_usage"]["prompt_tokens"],
            "completion_tokens": response.response_metadata["token_usage"]["completion_tokens"],
            "total_tokens": response.response_metadata["token_usage"]["total_tokens"],
        }
        
        # Local model, so cost is 0
        cost = 0.0
        
        return {
            "content": content,
            "usage": usage,
            "cost": cost,
        }


class ClaudeModel(BaseLLMModel):
    """Anthropic Claude API model"""
    
    def _setup(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library is not installed. pip install anthropic")
        
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("Claude API key is required.")
        
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Process messages for Claude (handles system messages and multimodal)
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            images = msg.get("images", [])
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
            else:
                # Process multimodal content for Claude
                if images:
                    # Claude supports images in content blocks
                    content_blocks = []
                    if isinstance(content, str) and content.strip():
                        content_blocks.append({"type": "text", "text": content})
                    
                    for img in images:
                        img_url = image_to_base64_url(img)
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_url.split(",")[1]  # Remove data:image/png;base64, prefix
                            }
                        })
                    
                    conversation_messages.append({
                        "role": role,
                        "content": content_blocks
                    })
                elif isinstance(content, list):
                    # Already formatted content
                    conversation_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    conversation_messages.append({
                        "role": role,
                        "content": content
                    })
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=conversation_messages,
        )
        
        # Claude response content may be a list
        content = ""
        if response.content:
            if isinstance(response.content[0], dict):
                content = response.content[0].get("text", "")
            else:
                content = str(response.content[0])
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        
        cost = self._calculate_cost(usage)
        
        return {
            "content": content,
            "usage": usage,
            "cost": cost,
        }
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate Claude cost"""
        # Claude 3 model pricing (example)
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015
        return (usage["prompt_tokens"] * input_cost_per_1k / 1000 + 
                usage["completion_tokens"] * output_cost_per_1k / 1000)


class GeminiModel(BaseLLMModel):
    """Google Gemini API model"""
    
    def _setup(self):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai library is not installed. pip install google-genai")
        
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        # New google-genai SDK uses Client instead of configure
        try:
            # Try new SDK approach with Client
            self.client = genai.Client(api_key=api_key)
            self.model_name_for_api = self.model_name
        except (AttributeError, TypeError):
            # Fallback to old approach if Client doesn't exist
            try:
                genai.configure(api_key=api_key)
                self.client = None
                self.model = genai.GenerativeModel(self.model_name)
            except AttributeError:
                raise ImportError("google-genai library is not properly installed or version is incompatible")
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Process messages for Gemini (supports multimodal)
        # Gemini API expects a list of message dictionaries with 'role' and 'parts'
        processed_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            images = msg.get("images", [])
            
            # Convert role to Gemini format (user -> user, assistant -> model, system -> user with note)
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                role = "user"
            elif role == "assistant":
                role = "model"
            
            parts = []
            
            # Add text content
            if isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, list):
                # Handle pre-formatted content list
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Handle image_url format
                            img_url = item.get("image_url", {})
                            if isinstance(img_url, dict):
                                img_url = img_url.get("url", "")
                            # Try to load image from path if it's a file path
                            if img_url.startswith("data:image"):
                                # Base64 image - decode it
                                import base64
                                header, encoded = img_url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                import PIL.Image as PILImage
                                pil_image = PILImage.open(io.BytesIO(img_bytes))
                                parts.append(pil_image)
                            elif not img_url.startswith("http"):
                                # Assume it's a file path
                                img_path = Path(img_url)
                                if img_path.exists():
                                    import PIL.Image as PILImage
                                    pil_image = PILImage.open(img_path)
                                    parts.append(pil_image)
                    else:
                        parts.append(str(item))
            
            # Add images if present
            if images:
                for img in images:
                    img_bytes = None
                    if isinstance(img, str):
                        img_path = Path(img)
                        if img_path.exists():
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()
                    elif isinstance(img, Image.Image):
                        if img.mode in ("RGBA", "LA"):
                            img = img.convert("RGB")
                        with io.BytesIO() as buffer:
                            img.save(buffer, format="PNG")
                            img_bytes = buffer.getvalue()
                    elif isinstance(img, bytes):
                        img_bytes = img
                    
                    if img_bytes:
                        import PIL.Image as PILImage
                        pil_image = PILImage.open(io.BytesIO(img_bytes))
                        parts.append(pil_image)
            
            if parts:
                processed_messages.append({
                    "role": role,
                    "parts": parts
                })
        
        # If no messages processed, return error
        if not processed_messages:
            raise ValueError("No valid messages to process for Gemini")
        
        # Try new SDK approach first, fallback to old approach
        try:
            # New google-genai SDK approach
            if hasattr(self, 'client') and self.client is not None:
                # Use new Client API
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                
                # New SDK: pass all messages as contents (no separate history parameter)
                # Convert processed_messages to the format expected by new SDK
                all_contents = []
                for msg in processed_messages:
                    all_contents.extend(msg["parts"])
                
                response = self.client.models.generate_content(
                    model=self.model_name_for_api,
                    contents=all_contents,
                    config=generation_config,
                )
            else:
                # Old SDK approach (fallback)
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                if len(processed_messages) > 1:
                    chat = self.model.start_chat(history=processed_messages[:-1])
                    response = chat.send_message(
                        processed_messages[-1]["parts"],
                        generation_config=generation_config,
                    )
                else:
                    response = self.model.generate_content(
                        processed_messages[0]["parts"],
                        generation_config=generation_config,
                    )
        except (AttributeError, TypeError) as e:
            # If new SDK approach fails, try old approach
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                if len(processed_messages) > 1:
                    chat = self.model.start_chat(history=processed_messages[:-1])
                    response = chat.send_message(
                        processed_messages[-1]["parts"],
                        generation_config=generation_config,
                    )
                else:
                    response = self.model.generate_content(
                        processed_messages[0]["parts"],
                        generation_config=generation_config,
                    )
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to generate content with Gemini API. New SDK error: {e}, Fallback error: {fallback_error}")
        
        # Extract content from response (handle both new and old SDK formats)
        try:
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'content'):
                # New SDK might return content differently
                if isinstance(response.content, str):
                    content = response.content
                elif hasattr(response.content, 'text'):
                    content = response.content.text
                else:
                    content = str(response.content)
            else:
                content = str(response)
        except Exception:
            content = str(response)
        
        # Gemini may require fetching token usage information separately
        # Should use count_tokens method in practice
        # Approximate token usage by counting words in processed messages
        total_text = " ".join([str(msg.get("parts", [])) for msg in processed_messages])
        
        # Try to get usage from response if available
        try:
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
            else:
                # Fallback to approximation
                usage = {
                    "prompt_tokens": len(total_text.split()),  # Approximation
                    "completion_tokens": len(content.split()) if content else 0,  # Approximation
                    "total_tokens": len(total_text.split()) + (len(content.split()) if content else 0),
                }
        except Exception:
            # Fallback to approximation
            usage = {
                "prompt_tokens": len(total_text.split()),  # Approximation
                "completion_tokens": len(content.split()) if content else 0,  # Approximation
                "total_tokens": len(total_text.split()) + (len(content.split()) if content else 0),
            }
        
        cost = self._calculate_cost(usage)
        
        return {
            "content": content,
            "usage": usage,
            "cost": cost,
        }
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate Gemini cost"""
        input_cost_per_1k = 0.00025
        output_cost_per_1k = 0.0005
        return (usage["prompt_tokens"] * input_cost_per_1k / 1000 + 
                usage["completion_tokens"] * output_cost_per_1k / 1000)


def create_model(provider: str, config: Dict[str, Any]) -> BaseLLMModel:
    """
    Create appropriate model instance based on provider
    
    Args:
        provider: "openai", "openrouter", "vllm", "claude", "gemini"
        config: Model configuration dictionary
        
    Returns:
        BaseLLMModel instance
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIModel(config)
    elif provider == "openrouter":
        return OpenRouterModel(config)
    elif provider == "vllm":
        return VLLMModel(config)
    elif provider == "claude":
        return ClaudeModel(config)
    elif provider == "gemini":
        return GeminiModel(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}. "
                        f"Supported providers: openai, openrouter, vllm, claude, gemini")

