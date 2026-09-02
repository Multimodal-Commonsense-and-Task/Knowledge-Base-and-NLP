"""
EMBGuard class for safety guardrail evaluation
Handles message construction and response parsing for guardrail tasks
"""
import re
import json
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from src.models import create_model


class EMBGuard:
    """
    EMBGuard class for safety guardrail evaluation
    Handles message construction and response parsing for guardrail tasks
    """
    
    def __init__(
        self,
        provider: str,
        model_config: Dict[str, Any],
    ):
        """
        Initialize EMBGuard with a model
        
        Args:
            provider: LLM provider ("openai", "openrouter", "vllm", "claude", "gemini")
            model_config: Model configuration dictionary
        """
        self.provider = provider
        self.model_config = model_config
        self.model = create_model(provider, model_config)
        
        # Import guardrail prompts
        try:
            from src.guardrail.prompts.guardrail_prompt import (
                GUARDRAIL_SYSTEM_PROMPT,
                GUARDRAIL_USER_PROMPT,
                GUARDRAIL_USER_PROMPT_WITH_THINKING,
                format_few_shot_examples,
                get_few_shot_messages
            )
            self.GUARDRAIL_SYSTEM_PROMPT = GUARDRAIL_SYSTEM_PROMPT
            self.GUARDRAIL_USER_PROMPT = GUARDRAIL_USER_PROMPT
            self.GUARDRAIL_USER_PROMPT_WITH_THINKING = GUARDRAIL_USER_PROMPT_WITH_THINKING
            self.format_few_shot_examples = format_few_shot_examples
            self.get_few_shot_messages = get_few_shot_messages
        except ImportError as e:
            raise ImportError(f"Failed to import guardrail prompts: {e}")
    
    def prepare_messages(
        self,
        action: str,
        image: Optional[Union[str, Path, "Image.Image", bytes]] = None,
        use_few_shot: bool = True,
        use_thinking: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for guardrail evaluation
        
        Args:
            action: Action to be evaluated
            image: Image input (optional, for multimodal evaluation). Can be:
                - str: Path to image file
                - Path: Path object to image file
                - PIL.Image.Image: PIL Image object
                - bytes: Image bytes
            use_few_shot: Whether to include few-shot examples
            use_thinking: Whether to use thinking mode (step-by-step reasoning)
            
        Returns:
            List of messages ready for model inference
        """
        if not action or not action.strip():
            raise ValueError("Action cannot be empty")
        
        # Select user prompt based on thinking mode
        if use_thinking:
            user_prompt = self.GUARDRAIL_USER_PROMPT_WITH_THINKING.format(action=action)
        else:
            user_prompt = self.GUARDRAIL_USER_PROMPT.format(action=action)
        
        # Construct messages
        messages = [
            {
                "role": "system",
                "content": self.GUARDRAIL_SYSTEM_PROMPT
            }
        ]
        
        # Add few-shot examples as conversation messages if requested
        if use_few_shot:
            few_shot_messages = self.get_few_shot_messages()
            messages.extend(few_shot_messages)
        
        # Add the actual user prompt
        user_message = {
            "role": "user",
            "content": user_prompt,
        }
        
        # Add image if provided
        if image is not None:
            # Validate image path if it's a string/Path
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
            user_message["images"] = [image]
        
        messages.append(user_message)
        
        return messages
    
    def evaluate(
        self,
        action: str,
        image: Optional[Union[str, Path, "Image.Image", bytes]] = None,
        use_few_shot: bool = True,
        use_thinking: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate an action for safety using guardrail
        
        Args:
            action: Action to be evaluated
            image: Image input (optional, for multimodal evaluation). Can be:
                - str: Path to image file
                - Path: Path object to image file
                - PIL.Image.Image: PIL Image object
                - bytes: Image bytes
            use_few_shot: Whether to include few-shot examples
            use_thinking: Whether to use thinking mode (step-by-step reasoning)
            **kwargs: Additional parameters for model generation (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary containing:
                - response: Raw response from model
                - parsed_response: Parsed response with potential_risk, risk_type, hazard, thinking
                - usage: Token usage information
                - cost: Cost in USD
        """
        # Prepare messages
        messages = self.prepare_messages(
            action=action,
            image=image,
            use_few_shot=use_few_shot,
            use_thinking=use_thinking
        )
        
        # Generate response
        response = self.model.generate_with_retry(messages, **kwargs)
        
        # Ensure content is a string (handle None or other types)
        response_content = response.get("content", "")
        if response_content is None:
            response_content = ""
        elif not isinstance(response_content, str):
            response_content = str(response_content)
        
        # Parse response
        parsed_response = self.parse_response(response_content, use_thinking=use_thinking)
        
        return {
            "response": response["content"],
            "parsed_response": parsed_response,
            "usage": response["usage"],
            "cost": response["cost"],
        }
    
    def parse_response(self, response_content: str, use_thinking: bool = False) -> Dict[str, Any]:
        """
        Parse model response to extract structured information
        
        Args:
            response_content: Raw response from model
            use_thinking: Whether thinking mode was used (to extract thinking content)
            
        Returns:
            Parsed response dictionary with potential_risk, risk_type, hazard, and optionally thinking
        """
        parsed = {
            "potential_risk": None,
            "risk_type": None,
            "hazard": None,
            "raw_response": response_content
        }
        
        # Ensure response_content is a string
        if response_content is None:
            response_content = ""
        elif not isinstance(response_content, str):
            response_content = str(response_content)
        
        # Extract thinking content if thinking mode was used
        if use_thinking and response_content:
            # Extract thinking content from <thinking> tags
            try:
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_content, re.DOTALL | re.IGNORECASE)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                else:
                    thinking_content = None
            except (TypeError, AttributeError):
                thinking_content = None
            
            parsed["thinking"] = thinking_content
        
        # Try to extract JSON from response
        if response_content:
            try:
                # Look for JSON block in response
                json_match = re.search(r'\{[^{}]*"potential_risk"[^{}]*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    parsed.update(parsed_json)
            except (json.JSONDecodeError, AttributeError, TypeError):
                # If JSON parsing fails, try to extract fields using regex
                try:
                    risk_match = re.search(r'potential_risk["\s:]*([^\n,}]+)', response_content, re.IGNORECASE)
                    if risk_match:
                        parsed["potential_risk"] = risk_match.group(1).strip().strip('"').strip("'")
                    
                    risk_type_match = re.search(r'risk_type["\s:]*([^\n,}]+)', response_content, re.IGNORECASE)
                    if risk_type_match:
                        parsed["risk_type"] = risk_type_match.group(1).strip().strip('"').strip("'")
                    
                    hazard_match = re.search(r'hazard["\s:]*([^\n,}]+)', response_content, re.IGNORECASE)
                    if hazard_match:
                        parsed["hazard"] = hazard_match.group(1).strip().strip('"').strip("'")
                except (TypeError, AttributeError):
                    pass
        
        return parsed
    
    def generate_with_retry(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate response with retry logic (delegates to underlying model)
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        return self.model.generate_with_retry(messages, **kwargs)
    
    def inference(
        self,
        action: str,
        image: Optional[Union[str, Path, "Image.Image", bytes]] = None,
        caution: Optional[str] = None,
        use_few_shot: bool = True,
        use_thinking: bool = False,
    ) -> Dict[str, str]:
        """
        Perform guardrail inference (compatible with BaseGuardrailModel interface)
        
        Args:
            action: Action description to evaluate
            image: Image input (optional). Can be:
                - str: Path to image file
                - Path: Path object to image file
                - PIL.Image.Image: PIL Image object
                - bytes: Image bytes
            caution: Safety caution (optional, currently unused)
            use_few_shot: Whether to include few-shot examples
            use_thinking: Whether to use thinking mode
            
        Returns:
            Dict[str, str]: {
                'potential_risk': 'risky' or 'benign',
                'risk_type': risk type or 'none',
                'hazard': hazard description or 'none'
            }
        """
        # Use evaluate method
        result_dict = self.evaluate(
            action=action,
            image=image,
            use_few_shot=use_few_shot,
            use_thinking=use_thinking
        )
        
        # Extract parsed_response
        parsed_response = result_dict.get('parsed_response', {})
        
        # Format result to match BaseGuardrailModel interface
        result = {
            'potential_risk': parsed_response.get('potential_risk', 'benign'),
            'risk_type': parsed_response.get('risk_type', 'none'),
            'hazard': parsed_response.get('hazard', 'none')
        }
        
        # Normalize potential_risk values ('safe' -> 'benign', 'risky'/'unsafe' -> 'risky')
        potential_risk = result.get('potential_risk', 'benign').lower()
        if potential_risk == 'safe':
            result['potential_risk'] = 'benign'
        elif potential_risk in ['unsafe', 'risky']:
            result['potential_risk'] = 'risky'
        else:
            result['potential_risk'] = potential_risk
        
        return result

