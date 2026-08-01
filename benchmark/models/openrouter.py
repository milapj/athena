import os
from openai import OpenAI
from model.openai import OpenAIModel
from benchmark.config import OPENROUTER_BASE_URL


class OpenRouterModel(OpenAIModel):
    """
    OpenRouter-backed model. Inherits retry logic and cost tracking from OpenAIModel.
    OpenRouter uses an OpenAI-compatible API, so we just override the client base_url.
    """

    def __init__(
        self,
        engine: str,
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        api_max_attempts: int = 10,
        gpt_waittime: int = 10,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            engine=engine,
            api_endpoint="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            api_max_attempts=api_max_attempts,
            gpt_waittime=gpt_waittime,
            **kwargs,
        )
        # Override the client to point at OpenRouter
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/grayscope/athena",
                "X-Title": "Athena Benchmark",
            },
        )
