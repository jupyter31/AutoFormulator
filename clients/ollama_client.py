"""
Ollama LLM Client

Native Ollama API client for local model inference with streaming support.
Uses Ollama's /api/chat endpoint for better model-specific feature supp            response = requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                stream=True,
                timeout=self.timeout
            ) as response:
                # Capture error details before raise_for_status
                if response.status_code != 200:
                    error_text = response.text[:1000]
                    logger.error(f"Ollama returned {response.status_code}: {error_text}")
                response.raise_for_status()
Supports:
- Streaming responses for real-time feedback (reduces perceived latency)
- DeepSeek R1: thinking mode for CoT capture
- Phi-4-reasoning: <think> tag extraction during streaming
- All other Ollama models

Usage:
    from clients.ollama_client import OllamaClient
    
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_id="deepseek-r1:8b",
        timeout_s=300  # Longer timeout for reasoning models
    )
    
    # With streaming progress callback
    def on_progress(channel, delta):
        print(f"[{channel}] {delta}", end="", flush=True)
    
    result = client.send_chat_request(
        model_name="deepseek-r1:8b",
        request={
            "messages": [...],
            "temperature": 0.7,
            "think": True,  # Enable DeepSeek thinking mode
            "on_delta": on_progress  # Optional progress callback
        }
    )
"""

from __future__ import annotations
import json
import logging
import re
import requests
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from .base_llm_client import BaseLLMClient, ChatResult

logger = logging.getLogger(__name__)

# Constants for <think> tag parsing
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class OllamaClient(BaseLLMClient):
    """
    Native Ollama API client using /api/chat endpoint with streaming support.
    
    Features:
    - Streaming responses for better UX with slow reasoning models
    - Automatic reasoning extraction from multiple sources
    - DeepSeek thinking mode support
    - Phi-4 <think> tag parsing during streaming
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "phi3:medium",
        timeout_s: int = 300,  # Longer default for reasoning models
        num_predict: Optional[int] = None  # Token limit for output
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model_id: Model identifier in Ollama (e.g., "deepseek-r1:8b", "phi4-reasoning:latest")
            timeout_s: Request timeout in seconds (default: 300 for reasoning models)
            num_predict: Maximum number of tokens to generate (limits output length)
        """
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout_s
        self.num_predict = num_predict
        
        logger.info(f"Initialized OllamaClient: {self.base_url}, model={self.model_id}, timeout={timeout_s}s")
    
    def send_chat_request(
        self,
        model_name: str,
        request: Dict[str, Any]
    ) -> ChatResult:
        """
        Send a chat request to Ollama with streaming support.
        
        Args:
            model_name: Model name (can override self.model_id)
            request: Request dict with keys:
                - messages: List[Dict[str, str]] with role/content
                - temperature: float (optional)
                - max_tokens: int (optional, mapped to num_predict)
                - think: bool (optional, DeepSeek R1 only - enables hidden thinking)
                - phi_tag_mode: bool (optional, forces <think> tag parsing during streaming)
                - on_delta: Callable[[str, str], None] (optional, progress callback)
                - options: Dict[str, Any] (optional, Ollama-specific options)
                - keep_alive: str (optional, keep model loaded duration)
                
        Returns:
            ChatResult dict with:
                - text: Final visible answer (no <think> tags)
                - reasoning_text: Captured CoT from thinking field or <think> tags
                - usage: Token counts and timing info
                - raw: Last JSON chunk from stream
                - flags: leak_think, has_reasoning
        """
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens")
        think_mode = request.get("think", False)
        phi_tag_mode = request.get("phi_tag_mode", False)
        on_delta = request.get("on_delta")
        extra_options = request.get("options", {})
        keep_alive = request.get("keep_alive")
        seed = request.get("seed")  # Extract seed if provided
        use_streaming = request.get("stream", False)  # Default to False for better reasoning capture
        
        # Use provided model_name or fall back to instance default
        model_to_use = model_name or self.model_id
        
        # Auto-detect phi_tag_mode for known phi models
        if "phi" in model_to_use.lower() and "reasoning" in model_to_use.lower():
            phi_tag_mode = True
            logger.debug(f"Auto-enabled phi_tag_mode for {model_to_use}")
        
        # Build Ollama API request
        body = {
            "model": model_to_use,
            "messages": messages,
            "stream": use_streaming,  # Configurable streaming mode
            "options": {
                "temperature": temperature,
                **extra_options  # Allow custom options
            }
        }
        
        # Add seed if specified (Ollama requires it in options)
        if seed is not None:
            body["options"]["seed"] = seed
        
        # Add max_tokens if specified (Ollama uses num_predict)
        if max_tokens:
            body["options"]["num_predict"] = max_tokens
        # Or use instance-level num_predict if set and max_tokens not provided
        elif self.num_predict is not None:
            body["options"]["num_predict"] = self.num_predict
        
        # DeepSeek R1: Enable thinking mode for CoT capture
        if think_mode:
            body["think"] = True
            logger.debug(f"Enabled thinking mode for {model_to_use}")
        
        # Keep model loaded
        if keep_alive:
            body["keep_alive"] = keep_alive
        
        # Non-streaming path (simpler, direct response)
        if not use_streaming:
            logger.debug(f"Non-streaming request to {self.base_url}/api/chat")
            logger.debug(f"Model: {model_to_use}, Messages: {len(messages)}, Think: {think_mode}")
            
            t0 = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=body,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                elapsed = time.time() - t0
                
                # Extract message content
                msg = result.get("message", {}) or {}
                visible_text = msg.get("content", "")
                reasoning_text = msg.get("thinking", "").strip() if think_mode else ""
                
                # ALWAYS extract and remove <think> tags from visible_text
                # These tags should never appear in the user-facing response
                if _THINK_OPEN in visible_text:
                    captured, stripped = self._extract_and_strip_think(visible_text)
                    if captured:
                        reasoning_text = (reasoning_text + "\n" + captured).strip() if reasoning_text else captured
                        visible_text = stripped
                        logger.warning(f"⚠️  <think> tags found in non-streaming response - extracted {len(captured)} chars")
                
                # Build usage info
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                    "total_duration_ms": result.get("total_duration", 0) // 1_000_000,
                }
                
                # Check if <think> tags remain after cleaning (should not happen)
                leak_think = _THINK_OPEN in visible_text or _THINK_CLOSE in visible_text
                
                return {
                    "text": visible_text,
                    "reasoning_text": reasoning_text or None,
                    "usage": usage,
                    "raw": result,
                    "process_tokens": None,
                    "flags": {
                        "leak_think": leak_think,
                        "has_reasoning": bool(reasoning_text),
                    }
                }
            
            except requests.exceptions.Timeout:
                elapsed = time.time() - t0
                logger.error(f"Non-streaming request timed out after {elapsed:.1f}s (limit: {self.timeout}s)")
                raise TimeoutError(f"Ollama request timed out after {elapsed:.1f}s")
            except requests.exceptions.RequestException as e:
                logger.error(f"Non-streaming request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"Ollama error details: {error_detail}")
                    except:
                        logger.error(f"Ollama response text: {e.response.text[:500]}")
                raise RuntimeError(f"Ollama request failed: {e}")
        
        # Streaming path (original implementation)
        logger.debug(f"Streaming from {self.base_url}/api/chat")
        logger.debug(f"Model: {model_to_use}, Messages: {len(messages)}, Think: {think_mode}, Phi: {phi_tag_mode}")
        logger.info(f"Request body keys: {list(body.keys())}")
        logger.info(f"Request model: {body['model']}")
        logger.info(f"Request options: {body['options']}")
        if messages:
            logger.info(f"First message: {messages[0]}")
            logger.info(f"First message keys: {list(messages[0].keys())}")
        
        t0 = time.time()
        visible_chunks = []
        thinking_chunks = []
        done_obj = {}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                stream=True,
                timeout=self.timeout
            )
            
            # Check status and capture error details before streaming
            if response.status_code != 200:
                error_text = response.text[:1000]
                logger.error(f"Ollama returned {response.status_code}: {error_text}")
                response.raise_for_status()
            
            with response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {line[:100]}")
                        continue
                    
                    msg = chunk.get("message", {}) or {}
                    
                    # Extract thinking content (DeepSeek R1 with think=True)
                    if think_mode and "thinking" in msg:
                        thinking_delta = msg.get("thinking", "")
                        if thinking_delta:
                            thinking_chunks.append(thinking_delta)
                            if on_delta:
                                on_delta("thinking", thinking_delta)
                    
                    # Extract visible content
                    content_delta = msg.get("content") or chunk.get("response") or ""
                    if content_delta:
                        if phi_tag_mode:
                            # Parse <think> tags during streaming
                            think_part, visible_part = self._split_think_delta(content_delta)
                            if think_part:
                                thinking_chunks.append(think_part)
                                if on_delta:
                                    on_delta("thinking", think_part)
                            if visible_part:
                                visible_chunks.append(visible_part)
                                if on_delta:
                                    on_delta("content", visible_part)
                        else:
                            visible_chunks.append(content_delta)
                            if on_delta:
                                on_delta("content", content_delta)
                    
                    # Check if this is the final chunk (after processing its message)
                    if chunk.get("done"):
                        done_obj = chunk
                        break
        
        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            logger.error(f"Request timed out after {elapsed:.1f}s (limit: {self.timeout}s)")
            raise TimeoutError(f"Ollama request timed out after {elapsed:.1f}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            # Try to get more details from the response
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Ollama error details: {error_detail}")
                except:
                    logger.error(f"Ollama response text: {e.response.text[:500]}")
            raise RuntimeError(f"Ollama request failed: {e}")
        
        elapsed = time.time() - t0
        
        # Assemble final text and reasoning
        visible_text = "".join(visible_chunks)
        reasoning_text = "".join(thinking_chunks).strip()
        
        logger.info(f"Captured thinking: {len(reasoning_text)} chars, visible: {len(visible_text)} chars")
        if reasoning_text:
            logger.debug(f"Thinking preview: {reasoning_text[:200]}...")
        
        # Post-processing: ALWAYS extract and remove any <think> tags from visible_text
        # These tags should never appear in the user-facing response
        if _THINK_OPEN in visible_text:
            captured, stripped = self._extract_and_strip_think(visible_text)
            if captured:
                # Add captured reasoning to reasoning_text if not already captured during streaming
                reasoning_text = (reasoning_text + "\n" + captured).strip() if reasoning_text else captured
                visible_text = stripped
                logger.debug(f"Post-processed <think> tags: {len(captured)} chars extracted, {len(stripped)} chars remaining")
                logger.warning(f"⚠️  <think> tags found in visible text - extracted {len(captured)} chars to reasoning_text")
        
        # Build usage info from done chunk
        usage = {
            "prompt_tokens": done_obj.get("prompt_eval_count", 0),
            "completion_tokens": done_obj.get("eval_count", 0),
            "total_tokens": done_obj.get("prompt_eval_count", 0) + done_obj.get("eval_count", 0),
            "total_duration_ms": done_obj.get("total_duration", 0) // 1_000_000,  # Convert ns to ms
            "load_duration_ms": done_obj.get("load_duration", 0) // 1_000_000,
            "elapsed_s": elapsed
        }
        
        # Check for reasoning leakage
        leak_think = (_THINK_OPEN in visible_text) or (_THINK_CLOSE in visible_text)
        if leak_think:
            logger.warning("⚠️  Reasoning leaked into final answer (found <think> tags in text)")
        
        # Build ChatResult
        result = {
            "text": visible_text.strip(),
            "usage": usage,
            "raw": done_obj,
            "reasoning_text": reasoning_text or None,
            "process_tokens": None,  # Ollama doesn't expose this separately
            "flags": {
                "leak_think": leak_think,
                "has_reasoning": bool(reasoning_text),
            }
        }
        
        logger.info(
            f"✓ Response: {len(visible_text)} chars, "
            f"reasoning: {len(reasoning_text or '')} chars, "
            f"time: {elapsed:.1f}s"
        )
        
        return result
    
    def _split_think_delta(self, delta: str) -> Tuple[str, str]:
        """
        Given a content delta, return (think_part, visible_part) for incremental parsing.
        Handles <think>...</think> blocks within streaming chunks.
        """
        think_part = ""
        vis_part = delta
        
        # Extract complete <think>...</think> blocks
        match = re.search(r"<think>(.*?)</think>", delta, flags=re.DOTALL)
        if match:
            think_part = match.group(1)
            vis_part = delta.replace(match.group(0), "")
        
        return think_part, vis_part
    
    def _extract_and_strip_think(self, text: str) -> Tuple[str, str]:
        """
        Extract all <think>...</think> blocks and return (joined_think, text_without_blocks).
        Used for post-processing when models unexpectedly include <think> tags.
        """
        blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        stripped = re.sub(r"</?think>", "", text, flags=re.DOTALL).strip()
        joined_think = "\n\n".join(b.strip() for b in blocks if b.strip())
        return joined_think, stripped
    
    def send_batch_chat_request(
        self,
        model_name: str,
        batch_requests: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[str]:
        """
        Send multiple chat requests sequentially.
        
        Ollama doesn't support batch requests, so we process them one by one.
        """
        results = []
        total = len(batch_requests)
        
        logger.info(f"Processing {total} requests sequentially...")
        
        for i, request in enumerate(batch_requests, 1):
            logger.info(f"Request {i}/{total}")
            try:
                result = self.send_chat_request(model_name, request)
                results.append(result.get("text", ""))
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                results.append("")
        
        return results
    
    def send_stream_chat_completion_request(
        self,
        model_name: str,
        request_data: Dict[str, Any]
    ):
        """
        Generator-based streaming chat request.
        
        Yields chunks as they arrive from Ollama, allowing the caller
        to process tokens in real-time without waiting for completion.
        
        Yields:
            Dict with either:
                - {"delta": str} - Token/chunk of text
                - {"done": True, "result": ChatResult} - Final result with metadata
                
        Example:
            for chunk in client.send_stream_chat_completion_request(model, request):
                if "delta" in chunk:
                    print(chunk["delta"], end="", flush=True)
                elif chunk.get("done"):
                    result = chunk["result"]
                    print(f"\n\nTotal tokens: {result['usage']['total_tokens']}")
        """
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens")
        think_mode = request_data.get("think", False)
        phi_tag_mode = request_data.get("phi_tag_mode", False)
        extra_options = request_data.get("options", {})
        keep_alive = request_data.get("keep_alive")
        
        # Use provided model_name or fall back to instance default
        model_to_use = model_name or self.model_id
        
        # Auto-detect phi_tag_mode for known phi models
        if "phi" in model_to_use.lower() and "reasoning" in model_to_use.lower():
            phi_tag_mode = True
        
        # Build Ollama API request
        body = {
            "model": model_to_use,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                **extra_options
            }
        }
        
        if max_tokens:
            body["options"]["num_predict"] = max_tokens
        
        if think_mode:
            body["think"] = True
        
        if keep_alive:
            body["keep_alive"] = keep_alive
        
        logger.debug(f"Generator streaming from {self.base_url}/api/chat")
        
        t0 = time.time()
        visible_chunks = []
        thinking_chunks = []
        done_obj = {}
        
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                stream=True,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Check if this is the final chunk
                    if chunk.get("done"):
                        done_obj = chunk
                        break
                    
                    msg = chunk.get("message", {}) or {}
                    
                    # Extract thinking content (DeepSeek R1)
                    if think_mode and "thinking" in msg:
                        thinking_delta = msg.get("thinking", "")
                        if thinking_delta:
                            thinking_chunks.append(thinking_delta)
                            # Yield thinking delta
                            yield {
                                "delta": thinking_delta,
                                "channel": "thinking"
                            }
                    
                    # Extract visible content
                    content_delta = msg.get("content") or chunk.get("response") or ""
                    if content_delta:
                        if phi_tag_mode:
                            # Parse <think> tags during streaming
                            think_part, visible_part = self._split_think_delta(content_delta)
                            if think_part:
                                thinking_chunks.append(think_part)
                                yield {
                                    "delta": think_part,
                                    "channel": "thinking"
                                }
                            if visible_part:
                                visible_chunks.append(visible_part)
                                yield {
                                    "delta": visible_part,
                                    "channel": "content"
                                }
                        else:
                            visible_chunks.append(content_delta)
                            yield {
                                "delta": content_delta,
                                "channel": "content"
                            }
        
        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            logger.error(f"Generator stream timed out after {elapsed:.1f}s")
            raise TimeoutError(f"Ollama request timed out after {elapsed:.1f}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Generator stream failed: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")
        
        elapsed = time.time() - t0
        
        # Assemble final result
        visible_text = "".join(visible_chunks)
        reasoning_text = "".join(thinking_chunks).strip()
        
        # Post-processing: ALWAYS extract and remove any <think> tags from visible_text
        # These tags should never appear in the user-facing response
        if _THINK_OPEN in visible_text:
            captured, stripped = self._extract_and_strip_think(visible_text)
            if captured:
                reasoning_text = (reasoning_text + "\n" + captured).strip() if reasoning_text else captured
                visible_text = stripped
                logger.warning(f"⚠️  <think> tags found in generator stream - extracted {len(captured)} chars")
        
        # Build usage info
        usage = {
            "prompt_tokens": done_obj.get("prompt_eval_count", 0),
            "completion_tokens": done_obj.get("eval_count", 0),
            "total_tokens": done_obj.get("prompt_eval_count", 0) + done_obj.get("eval_count", 0),
            "total_duration_ms": done_obj.get("total_duration", 0) // 1_000_000,
            "load_duration_ms": done_obj.get("load_duration", 0) // 1_000_000,
            "elapsed_s": elapsed
        }
        
        # Check if <think> tags remain after cleaning (should not happen)
        leak_think = (_THINK_OPEN in visible_text) or (_THINK_CLOSE in visible_text)
        
        result = {
            "text": visible_text.strip(),
            "usage": usage,
            "raw": done_obj,
            "reasoning_text": reasoning_text or None,
            "process_tokens": None,
            "flags": {
                "leak_think": leak_think,
                "has_reasoning": bool(reasoning_text),
            }
        }
        
        # Yield final result
        yield {
            "done": True,
            "result": result
        }
