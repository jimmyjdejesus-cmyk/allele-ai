from typing import Any

class APIError(Exception):
    code: Any
    type: Any

class APITimeoutError(Exception): ...

class AuthenticationError(Exception): ...

class RateLimitError(Exception):
    retry_after: int


class ModelsListResponse:  # minimal container
    data: Any


class ModelsAPI:
    async def list(self) -> ModelsListResponse: ...


class CompletionsAPI:
    async def create(self, *args: Any, **kwargs: Any) -> Any: ...


class ChatAPI:
    completions: CompletionsAPI


class AsyncOpenAI:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int | None = None, max_retries: int | None = None) -> None: ...

    models: ModelsAPI
    chat: ChatAPI

    async def close(self) -> None: ...
