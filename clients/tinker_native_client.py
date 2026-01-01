"""
Native Tinker SDK client for base models and inference.

Uses the tinker Python package directly instead of OpenAI-compatible endpoint.
"""
from typing import Any, Dict, List, Optional
import os
import logging
from .base_llm_client import BaseLLMClient, ChatResult

logger = logging.getLogger(__name__)


class TinkerNativeClient(BaseLLMClient):
    """Client for Tinker base models using native SDK."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B-Base",
        api_key: Optional[str] = None,
        timeout_s: int = 300,
    ):
        """
        Initialize Tinker native client.
        
        Args:
            model_id: Model identifier from Tinker catalog (e.g., "Qwen/Qwen3-8B-Base")
            api_key: Tinker API key, defaults to TINKER_API_KEY env var
            timeout_s: Request timeout in seconds
        """
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "Native Tinker client requires 'tinker' package. "
                "Install with: pip install tinker-ai"
            )
        
        # Get API key
        key = api_key or os.getenv("TINKER_API_KEY")
        if not key:
            # Set it in environment for tinker SDK
            raise ValueError(
                "API key must be provided via TINKER_API_KEY environment variable"
            )
        
        if not os.getenv("TINKER_API_KEY"):
            os.environ["TINKER_API_KEY"] = key
        
        self.model_id = model_id
        self.timeout_s = timeout_s
        
        # Initialize Tinker service
        self.service = tinker.ServiceClient()
        self.model = None  # Lazy load
        
        logger.info(f"Initialized TinkerNativeClient: model={model_id}, timeout={timeout_s}s")

    def _get_model(self):
        """
        Lazy load the model using the training client trick.
        
        Creates a dummy LoRA training client to mount the base model,
        then gets a sampling client from it (before any training steps).
        This allows sampling from the frozen base weights.
        """
        if self.model is None:
            import tinker
            logger.info(f"Mounting base model {self.model_id} via training client...")
            
            # Create dummy training client to mount base model
            base_client = self.service.create_lora_training_client(
                base_model=self.model_id,
                rank=64  # Standard rank, not used since we won't train
            )
            
            # Get sampling client from the training client pointing to base model
            # Use "base" as model_path to sample from the frozen base weights
            self.model = base_client.create_sampling_client(model_path="base")
            logger.info(f"✓ Sampling client ready for {self.model_id}")
        
        return self.model

    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """
        Send a chat completion request using native Tinker SDK.
        
        Args:
            model_name: Ignored (uses self.model_id)
            request: Dict with:
                - messages: List[Dict[str, str]] with role/content
                - temperature: float (optional)
                - max_tokens: int (optional)
                - top_p: float (optional)
        
        Returns:
            ChatResult dict
        """
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 2048)
        top_p = request.get("top_p", 0.9)
        
        # Convert messages to prompt (Qwen3 base model expects raw text)
        prompt = self._messages_to_prompt(messages)
        
        logger.debug(f"Sending request to {self.model_id}, prompt length: {len(prompt)}")
        
        try:
            sampler = self._get_model()
            
            # Generate response using Tinker SDK's sample method
            response = sampler.sample(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Extract text from response
            if isinstance(response, dict):
                text = response.get("generated_text", "")
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)
            
            # Build result
            result: ChatResult = {
                "text": text.strip(),
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Rough estimate
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split())
                },
                "reasoning_text": None,
                "process_tokens": None,
                "flags": {},
                "raw": response if isinstance(response, dict) else {"text": response}
            }
            
            logger.info(f"✓ Generated {len(text)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Tinker SDK error: {e}")
            raise RuntimeError(f"Tinker inference failed: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string.
        
        For base models, we concatenate messages with simple formatting.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant: ")
        
        return "\n".join(prompt_parts)


__all__ = ["TinkerNativeClient"]
