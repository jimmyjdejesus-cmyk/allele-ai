from typing import Any, AsyncIterator

class APIError(Exception): ...

class APITimeoutError(Exception): ...

class AuthenticationError(Exception): ...

class RateLimitError(Exception):
    retry_after: int


class ModelsListResponse:  # minimal container
    data: Any


class AsyncOpenAI:
    class models:
        async def list(self) -> ModelsListResponse: ...

    class chat:
        class completions:
            async def create(self, *args, **kwargs) -> Any: ...

    async def close(self) -> None: ...
