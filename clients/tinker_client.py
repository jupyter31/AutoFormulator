"""Tinker LLM client using OpenAI SDK."""
from typing import Any, Dict, Optional
import os
import logging

from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error

logger = logging.getLogger(__name__)


class TinkerClient(BaseLLMClient):
    """Client for Tinker models using OpenAI SDK with API key authentication."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: int = 300,
    ):
        """
        Initialize Tinker client.
        
        Args:
            base_url: Tinker OpenAI compatible endpoint URL
                     Defaults to TINKER_BASE_URL env var or the standard Tinker prod URL
            model_name: Model path (e.g. tinker://...)
                       Defaults to TINKER_MODEL_NAME env var
            api_key: Tinker API key for authentication
                    Defaults to TINKER_API_KEY env var
            timeout_s: Request timeout in seconds (default: 300)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Tinker client requires 'openai' package. "
                "Install with: pip install openai"
            )
        
        self.base_url = base_url or os.getenv(
            "TINKER_BASE_URL", 
            "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
        )
        
        self.default_model_name = model_name or os.getenv("TINKER_MODEL_NAME")
        self.timeout_s = timeout_s
        
        # Get API key from parameter or environment
        key = api_key or os.getenv("TINKER_API_KEY")
        if not key:
            raise ValueError(
                "API key must be provided either as 'api_key' parameter or "
                "via TINKER_API_KEY environment variable"
            )
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=key,
            timeout=float(self.timeout_s)
        )
        
        logger.info(f"Initialized Tinker client with timeout={timeout_s}s, base_url={self.base_url}, model={self.default_model_name}")

    @retry_on_error(max_retries=3, initial_wait=30.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """Send a chat completion request with retry logic for 429/503."""
        
        # Build messages in OpenAI format
        messages = request.get("messages", [])
        
        # Determine model to use
        model = model_name or self.default_model_name
        if not model:
             raise ValueError("Model name must be provided either in init or request")

        # Build kwargs for the API call
        kwargs = {
            "model": model,
            "messages": messages,
        }
        
        # Add temperature if provided
        if "temperature" in request:
            kwargs["temperature"] = request["temperature"]
        
        # Handle token limits
        if "max_completion_tokens" in request:
            kwargs["max_tokens"] = request["max_completion_tokens"]
        elif "max_tokens" in request:
            kwargs["max_tokens"] = request["max_tokens"]
        
        # Add seed if provided
        if "seed" in request:
            kwargs["seed"] = request["seed"]
            
        # Add top_p if provided
        if "top_p" in request:
            kwargs["top_p"] = request["top_p"]

        # Send request
        logger.debug(f"Sending Tinker request to model={kwargs['model']}")
        response = self.client.chat.completions.create(**kwargs)
        
        # Extract response content
        choices = response.choices if hasattr(response, 'choices') else []
        if choices:
            message = choices[0].message
            content = message.content if hasattr(message, 'content') else ""
        else:
            content = ""
        
        # Extract usage information
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage_obj = response.usage
            if hasattr(usage_obj, 'prompt_tokens'):
                usage["prompt_tokens"] = usage_obj.prompt_tokens
            if hasattr(usage_obj, 'completion_tokens'):
                usage["completion_tokens"] = usage_obj.completion_tokens
            if hasattr(usage_obj, 'total_tokens'):
                usage["total_tokens"] = usage_obj.total_tokens
        
        # Return raw response
        result: ChatResult = {
            "text": (content or "").strip(),
            "usage": usage,
            "reasoning_text": None,
            "process_tokens": None,
            "flags": {},
            "raw": response.model_dump() if hasattr(response, 'model_dump') else {}
        }
        
        return result


__all__ = ["TinkerClient"]
