from typing import Any, AsyncIterator

class APIError(Exception):
    code: str | None
    type: str | None

class APITimeoutError(Exception):
    pass

class AuthenticationError(Exception):
    pass

class RateLimitError(Exception):
    retry_after: float | None

class ChatCompletion:  # Minimal placeholder for sync completion
    choices: Any
    usage: Any

class ChatCompletionChunk:  # Minimal placeholder for streamed chunk
    choices: Any
    usage: Any

class AsyncStream(AsyncIterator[ChatCompletionChunk]):
    def __aiter__(self) -> AsyncStream: ...
    async def __anext__(self) -> ChatCompletionChunk: ...

class AsyncOpenAI:
    def __init__(self, *args: Any, **kwargs: Any):
        ...

    async def close(self) -> None: ...
    @property
    def models(self) -> Any: ...
    @property
    def chat(self) -> Any: ...
