import os
import itertools
import time
from openai import OpenAI
import random

from datetime import timedelta
from typing import List, Dict, Union, Any
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from model.model import Model
# from athena import cache


class OpenAIModel(Model):
    """
    Wrapper for calling OpenAI with safety & retry loops + caching.

    Supports both completion (e.g., text-davinci-003) and chat (e.g., gpt-3.5-turbo) endpoints.
    """

    def __init__(
        self,
        api_key: str = None,
        engine: str = "gpt-4",
        api_max_attempts: int = 60,
        api_endpoint: str = "chat",   # "chat" or "completion"
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop_token: str = None,
        log_probs: int = 1,    # only used for completion endpoint
        num_samples: int = 1,
        echo: bool = True,     # only used for completion endpoint

        prompt_cost: float = None,
        completion_cost: float = None,
        gpt_waittime: int = 60
    ):
        """
        :param api_key: Your OpenAI API key. If not provided, we use OPENAI_API_KEY from environment.
        :param engine: The OpenAI model name (e.g. "gpt-4" or "gpt-3.5-turbo").
        :param api_max_attempts: How many times to retry on rate limits or transient errors.
        :param api_endpoint: "completion" or "chat".
        :param temperature: Sampling temperature (0-2).
        :param top_p: Top-p sampling (0-1).
        :param max_tokens: Maximum tokens to generate in the completion/response.
        :param stop_token: String or list of strings where the API will stop generating further tokens.
        :param log_probs: Number of top logprobs to return (only works for completion endpoint).
        :param num_samples: Number of parallel responses to generate (n=... in the API).
        :param echo: Whether to echo back the prompt in the response (only for completion endpoint).
        :param prompt_cost: Cost per token for input (optional, for accounting).
        :param completion_cost: Cost per token for output (optional, for accounting).
        :param gpt_waittime: Time to wait (in seconds) if you hit a rate-limit or transient error before retry.
        """

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        self.engine = engine
        self.api_max_attempts = api_max_attempts
        self.api_endpoint = api_endpoint.lower()

        self.max_tokens = max_tokens
        self.stop_token = stop_token
        self.log_probs = log_probs
        self.num_samples = num_samples
        self.echo = echo

        self.temperature = temperature
        self.top_p = top_p
        self.gpt_waittime = gpt_waittime

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0

    def __update_cost__(self, response):
        """
        If prompt_cost and completion_cost were provided,
        update the total cost from usage data in the API response.
        """
        if self.prompt_cost and self.completion_cost and hasattr(response, "usage"):
            usage = response.usage
            completion_tokens = getattr(usage, "completion_tokens", 0)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            cost = completion_tokens * self.completion_cost + prompt_tokens * self.prompt_cost
            self.total_cost += cost

    # @cache.cached(
    #     data_ex=timedelta(days=30),
    #     no_data_ex=timedelta(hours=1),
    #     prepended_key_attr=(
    #         "engine,num_samples,log_probs,echo,"
    #         "temperature=float(0),top_p=float(1.0),stop_token,max_tokens"
    #     ),
    # )
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        """
        Main public method for generating from the OpenAI API, with caching.
        Depending on api_endpoint, calls either completion or chat.
        """
        if self.api_endpoint == "completion":
            response = self.__safe_openai_completion_call__(prompt, *args, **kwargs)
        elif self.api_endpoint == "chat":
            response = self.__safe_openai_chat_call__(prompt, *args, **kwargs)
        else:
            raise ValueError(f"Unknown api_endpoint: {self.api_endpoint}. Use 'completion' or 'chat'.")

        self.__update_cost__(response)
        return response

    def __safe_openai_completion_call__(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        stop_token: str = None,
        logprobs: int = None,
        num_samples: int = None,
        echo: bool = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        Calls the OpenAI Completion endpoint with multiple retries on rate-limits or errors.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_token is None:
            stop_token = self.stop_token
        if logprobs is None:
            logprobs = self.log_probs
        if num_samples is None:
            num_samples = self.num_samples
        if echo is None:
            echo = self.echo

        last_exc = None
        for _ in range(self.api_max_attempts):
            try:
                response = self.client.completions.create(
                    model=self.engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    n=num_samples,
                    echo=echo,
                    stop=stop_token,
                )
                # Convert to dict format for compatibility
                return {
                    "choices": [
                        {
                            "message": {
                                "content": choice.text
                            }
                        } for choice in response.choices
                    ]
                }
            except Exception as e:
                last_exc = e
                print(f"[OpenAIModel] Completion error: {e}. Retrying after {self.gpt_waittime}s...")
                time.sleep(self.gpt_waittime + random.randint(1, 10))

        # Return a fallback response if all retries fail
        return {
            "API Error": True,
            "choices": [
                {
                    "message": {
                        "content": (
                            prompt + f" OPENAI Error - {last_exc}\n"
                            "This is a fallback response due to an error."
                        )
                    }
                }
            ]
        }

    def __safe_openai_chat_call__(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop_token: str = None,
        num_samples: int = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        Calls the OpenAI ChatCompletion endpoint with multiple retries on rate-limits or errors.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if stop_token is None:
            stop_token = self.stop_token
        if num_samples is None:
            num_samples = self.num_samples

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc = None
        for _ in range(self.api_max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=num_samples,
                    stop=stop_token,
                )
                # Convert to dict format for compatibility
                usage = getattr(response, "usage", None)
                result = {
                    "choices": [
                        {
                            "message": {
                                "content": choice.message.content
                            }
                        } for choice in response.choices
                    ]
                }
                if usage:
                    result["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    }
                return result
            except Exception as e:
                last_exc = e
                print(f"[OpenAIModel] Chat error: {e}. Retrying after {self.gpt_waittime}s...")
                time.sleep(self.gpt_waittime + random.randint(1, 10))

        # Return a fallback response if all retries fail
        return {
            "API Error": True,
            "choices": [
                {
                    "message": {
                        "content": (
                            prompt + f" OPENAI Error - {last_exc}\n"
                            "This is a fallback response due to an error."
                        )
                    }
                }
            ]
        }
