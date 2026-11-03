import os
import sys
import json
import logging
import time
import requests
from typing import Callable, Optional
from typing import Literal
from typing_extensions import TypeAlias

from anthropic import Anthropic
from openai import OpenAI
from ollama import Client
from dotenv import load_dotenv
from tabulate import tabulate
import instructor

import httpx
import io
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', None)

load_dotenv()
logger = logging.getLogger(__name__)

OPEN_API_KEY = ""
ANTHROPIC_API_KEY = ""
OLLAMA_BASE_URL_CLOUD = "https://ollama.com/v1/"
OLLAMA_BASE_URL_LOCAL = "http://127.0.0.1:11434/v1/"
OLLAMA_NATIVE_URL = "http://localhost:11434"      # for /api/show, /api/pull
OLLAMA_API_KEY = 

Message: TypeAlias = dict[Literal["role", "content"], str]
Messages: TypeAlias = list[Message]

MODEL_MAPPING = {
    "gpt": ["gpt-4o-mini",],
    "claude": ["claude-sonnet-4-6",],
    "ollama": ["qwen2.5:32b", "qwen2.5:14b", "qwen3.6:27b", "gemma4:26b", "gemma4:e4b", "gemini-3-flash-preview: latest", "gpt-oss:20b", "gpt-oss-safeguard:20b", "llama3.1:8b", "deepseek-v4-pro"],
}


class GenerateResponse:
    def __init__(
        self,
        model_name: str,
        backend: str,
        messages: Messages,
        api_key: Optional[str] = None,
        validation_model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        stop_sequences: Optional[list[str]] = None,
        verbose: bool = False,
        max_tries: int = 20,
        initial_sleep_time: float = 1.0,
        backoff_factor: float = 1.5,
    ):
        self.model_name = model_name
        self.backend = backend
        self.api_key = api_key
        self.validation_model = validation_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop_sequences or []
        self.verbose = verbose
        self.messages = messages
        self.max_tries = max_tries
        self.initial_sleep_time = initial_sleep_time
        self.backoff_factor = backoff_factor

        if backend == "gpt":
            self.base_client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY)
            self.client = instructor.from_provider(self.base_client)
            logger.info(f"OpenAI backend: {self.model_name}")

        elif backend == "claude":
            self.base_client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY)
            self.client = instructor.from_provider(self.base_client)
            logger.info(f"Anthropic backend: {self.model_name}")

        elif backend == "ollama":
            is_cloud = "cloud" in self.model_name
            
            ollama_base_url = OLLAMA_BASE_URL_CLOUD if is_cloud else (os.getenv("OLLAMA_BASE_URL_LOCAL") or OLLAMA_BASE_URL_LOCAL)
            ollama_api_key  = (self.api_key or os.getenv("OLLAMA_API_KEY") or OLLAMA_API_KEY) if is_cloud else "ollama"
            
            self.base_client = OpenAI(base_url=ollama_base_url, api_key=ollama_api_key)
            self._ensure_model_available(self.model_name)
            self.client = instructor.from_openai(self.base_client, mode=instructor.Mode.JSON)
            logger.info(f"Ollama backend ({'cloud' if 'ollama.com' in ollama_base_url else 'local'}): {self.model_name}")

        else:
            raise ValueError(f"Unsupported backend: '{backend}'. Choose from: gpt, claude, ollama.")

    def _ensure_model_available(self, model_name: str) -> None:
        """Ensure Ollama model is available locally, pulling it if necessary."""
        try:
            response = requests.post(f"{OLLAMA_NATIVE_URL}/api/show", json={"name": model_name}, timeout=5)
            if response.status_code == 200:
                logger.info(f"Model {model_name} already loaded")
                return

            logger.info(f"Pulling {model_name}...")
            pull_response = requests.post(f"{OLLAMA_NATIVE_URL}/api/pull", json={"name": model_name}, timeout=600)
            if pull_response.status_code != 200:
                raise RuntimeError(f"Failed to pull model {model_name}")

            logger.info(f"Model {model_name} ready")

        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise RuntimeError("Ollama server not running. Start with: ollama serve")

    def _call_api(self) -> str:
        """Make a single API call and return the raw string output."""
        if self.verbose:
            print(tabulate([m.values() for m in self.messages], ["role", "content"], "simple_grid", maxcolwidths=[50, 70],))

        if self.backend == "gpt":
            response = self.base_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                stop=self.stop or None,
            )
            return response.choices[0].message.content

        elif self.backend == "claude":
            has_system = self.messages[0]["role"] == "system"
            kwargs = {"system": self.messages[0]["content"]} if has_system else {}
            response = self.base_client.messages.create(
                model=self.model_name,
                messages=self.messages[1:] if has_system else self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop or None,
                **kwargs,
            )
            return response.content[0].text

        else:  # ollama
            response = self.base_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop or None,
            )
            return response.choices[0].message.content

    def generate_response_basic(self) -> str:
        """Call the API with exponential backoff retry on rate-limit errors."""
        return self.retry_and_backoff(self._call_api)()

    def retry_and_backoff(self, func: Callable) -> Callable:
        """Wrap a callable with retry logic and exponential backoff."""
        def wrapper(*args, **kwargs):
            sleep_time = self.initial_sleep_time
            last_exception = None

            for attempt in range(1, self.max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if "rate limit" in str(e).lower().replace("_", " "):
                        logger.warning(
                            f"Rate limit hit (attempt {attempt}/{self.max_tries}). "
                            f"Sleeping {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)
                        sleep_time *= self.backoff_factor
                    else:
                        raise

            raise RuntimeError(
                f"Maximum retries ({self.max_tries}) exceeded."
            ) from last_exception

        return wrapper
        
    