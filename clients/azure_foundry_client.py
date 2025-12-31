"""Azure OpenAI LLM client using OpenAI SDK."""
from typing import Any, Dict, Optional
import os
import logging
import re

from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error

logger = logging.getLogger(__name__)


class AzureFoundryClient(BaseLLMClient):
    """Client for Azure OpenAI models using OpenAI SDK with API key authentication."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout_s: int = 300,
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            endpoint: Azure OpenAI endpoint URL (can include full path or just base)
                     Defaults to AZURE_INFERENCE_SDK_ENDPOINT env var
            model_name: Model deployment name
                       Defaults to DEPLOYMENT_NAME env var
            api_key: Azure API key for authentication
                    Defaults to AZURE_API_KEY env var
            api_version: Azure OpenAI API version
                        Defaults to extracted from endpoint or "2024-12-01-preview"
            timeout_s: Request timeout in seconds (default: 300)
        """
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "Azure OpenAI client requires 'openai' package. "
                "Install with: pip install openai"
            )
        
        raw_endpoint = endpoint or os.getenv(
            "AZURE_INFERENCE_SDK_ENDPOINT"
        )
        
        # Extract base endpoint (remove /openai/deployments/... if present)
        if "/openai/deployments/" in raw_endpoint:
            match = re.match(r'(https://[^/]+)', raw_endpoint)
            if match:
                self.endpoint = match.group(1) + "/"
            else:
                self.endpoint = raw_endpoint
            
            # Extract API version from query string if present
            version_match = re.search(r'api-version=([^&]+)', raw_endpoint)
            if version_match and not api_version:
                api_version = version_match.group(1)
            
            logger.info(f"Detected Azure OpenAI URL, using base endpoint: {self.endpoint}")
        else:
            self.endpoint = raw_endpoint if raw_endpoint.endswith('/') else raw_endpoint + '/'
            
        self.default_model_name = model_name or os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = api_version or "2024-12-01-preview"
        self.timeout_s = timeout_s
        
        # Get API key from parameter or environment
        key = api_key or os.getenv("AZURE_API_KEY")
        if not key:
            raise ValueError(
                "API key must be provided either as 'api_key' parameter or "
                "via AZURE_API_KEY environment variable"
            )
        
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=key,
            timeout=float(self.timeout_s)
        )
        
        logger.info(f"Initialized AzureOpenAI client with timeout={timeout_s}s, endpoint={self.endpoint}, model={self.default_model_name}, api_version={self.api_version}")

    @retry_on_error(max_retries=3, initial_wait=30.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """Send a chat completion request with retry logic for 429/503."""
        
        # Build messages in OpenAI format (already compatible)
        messages = request.get("messages", [])
        
        # Build kwargs for the API call
        kwargs = {
            "model": model_name or self.default_model_name,
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
        
        # Send request
        logger.debug(f"Sending Azure OpenAI request to model={kwargs['model']}")
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
        }
        
        return result


__all__ = ["AzureFoundryClient"]
