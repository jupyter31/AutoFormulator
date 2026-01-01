#%%
import os
import time
import random
import json
from clients.azure_foundry_client import AzureFoundryClient
from clients.tinker_client import TinkerClient
from clients.ollama_client import OllamaClient

MAX_RETRY = 5

def _load_vscode_settings():
    """Load environment variables from VS Code settings if not already set."""
    settings_path = os.path.join(os.getcwd(), ".vscode", "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                # Remove comments from JSON
                content = f.read()
                lines = [line.split('//')[0] for line in content.split('\n')]
                clean_content = '\n'.join(lines)
                settings = json.loads(clean_content)
                
                # Set environment variables if not already set
                for key in ["TINKER_API_KEY", "AZURE_API_KEY", "AZURE_INFERENCE_SDK_ENDPOINT", "DEPLOYMENT_NAME"]:
                    if key in settings and not os.getenv(key):
                        os.environ[key] = settings[key]
        except Exception as e:
            pass  # Silently continue if settings can't be loaded

# Load settings at module import
_load_vscode_settings()

# Global client instances (lazy initialization)
_clients = {}

def _get_llm_client(client_type="azure"):
    """Get or create the global LLM client instance."""
    global _clients
    if client_type not in _clients:
        if client_type == "tinker":
            _clients[client_type] = TinkerClient(timeout_s=300)
        elif client_type == "ollama":
            # Get Ollama settings from VS Code settings or environment
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_id = os.getenv("OLLAMA_MODEL_ID", "phi3:medium")
            _clients[client_type] = OllamaClient(
                base_url=base_url,
                model_id=model_id,
                timeout_s=300
            )
        else:
            # Initialize with defaults from environment variables
            _clients[client_type] = AzureFoundryClient(
                timeout_s=300  # 5 minute timeout
            )
    return _clients[client_type]

def chat_gpt(
        user_prompt=None, 
        system_prompt=None, 
        n_used=1,
        logprobs=False,      # kept for API compatibility; not used
        seed=None,
        llm_name=None,       # kept for compatibility
        engine_used='gpt-4o'
    ):
    """
    Calls Azure Foundry LLM client using deployment names selected by `engine_used`.
    All secrets and endpoints are read from environment variables via the client.
    
    Returns a response object compatible with OpenAI's format for backward compatibility.
    """

    # Determine client and model based on engine_used
    if engine_used and (engine_used.startswith("tinker") or "tinker" in engine_used):
        client_type = "tinker"
        # If engine_used is a full tinker path, use it. Otherwise fallback to env var.
        if engine_used.startswith("tinker://"):
            model_name = engine_used
        else:
            model_name = os.getenv("TINKER_MODEL_NAME")
    elif engine_used and "ollama" in engine_used.lower():
        client_type = "ollama"
        # Extract model name from engine_used (e.g., "ollama:deepseek-r1:8b" or just "ollama")
        if ":" in engine_used:
            # Format: "ollama:model_name" or "ollama:deepseek-r1:8b"
            parts = engine_used.split(":", 1)
            model_name = parts[1] if len(parts) > 1 else os.getenv("OLLAMA_MODEL_ID", "phi3:medium")
        else:
            model_name = os.getenv("OLLAMA_MODEL_ID", "phi3:medium")
    else:
        client_type = "azure"
        # Use DEPLOYMENT_NAME from environment (set in VS Code settings)
        # This maps to your Azure OpenAI deployment (e.g., gpt-4o)
        model_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    # Build request dict for the client
    request = {
        "messages": messages if messages else [{"role": "user", "content": user_prompt or system_prompt}],
        "stream": True if client_type == "ollama" else False,  # Enable streaming for Ollama
    }
    
    if seed is not None:
        request["seed"] = seed

    # Get the client instance
    client = _get_llm_client(client_type)

    # --- simple retry with exponential backoff ---
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            # For n_used > 1, we need to make multiple requests
            # since Azure AI Inference doesn't support n parameter
            responses = []
            for _ in range(n_used):
                result = client.send_chat_request(model_name, request)
                responses.append(result)
            
            # Convert to OpenAI-compatible response format
            class Choice:
                def __init__(self, text, index):
                    self.message = type('Message', (), {'content': text})()
                    self.index = index
            
            class Response:
                def __init__(self, choices):
                    self.choices = choices
            
            choices = [Choice(resp['text'], i) for i, resp in enumerate(responses)]
            response = Response(choices)
            
            # Basic sanity check
            _ = response.choices[0].message.content  # will raise if missing
            return response
            
        except Exception as e:
            last_err = e
            # jittered backoff
            sleep_time = min(2.5, 1.0 + random.random()) * (1.7 ** attempt)
            time.sleep(sleep_time)

    # If we got here, all retries failed
    raise RuntimeError(f"chat_gpt failed after {MAX_RETRY} attempts. Last error: {last_err}")
