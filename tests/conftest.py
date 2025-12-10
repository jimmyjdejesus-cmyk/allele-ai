"""Pytest configuration and fixtures for Allele testing."""

import pytest
import asyncio
import aiohttp
import os
from typing import List, Optional


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def ollama_available_models() -> List[str]:
    """Get available Ollama models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return [model.get("name", "") for model in data.get("models", [])]
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Could not connect to Ollama to get available models: {e}")
    return []


@pytest.fixture(scope="session")
async def ensure_gemma_models(ollama_available_models):
    """Ensure gemma2:2b model is available for consistent testing."""
    test_model = "gemma2:2b"  # Standardized test model

    if test_model in ollama_available_models:
        return test_model

    # Try to pull the standardized test model
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/pull",
                json={"name": test_model},
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minute timeout for pulling
            ) as response:
                if response.status == 200:
                    # Pull successful
                    return test_model
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Failed to pull test model '{test_model}': {e}")

    return None  # Test model unavailable


@pytest.fixture
def skip_if_cloud_models_unavailable():
    """Skip test if cloud models are not configured."""
    cloud_api_key = os.getenv("OLLAMA_API_KEY", "")

    # Basic check - if no key is set, assume cloud not configured
    if not cloud_api_key:
        pytest.skip("Cloud models not configured (no OLLAMA_API_KEY)")


@pytest.fixture
async def local_ollama_config(ensure_gemma_models):
    """Configuration for local Ollama testing."""
    gemma_model = ensure_gemma_models
    if gemma_model and isinstance(gemma_model, str):
        from allele.agent import AgentConfig
        return AgentConfig(
            llm_provider="ollama",
            model_name=gemma_model,
            api_key="",  # Ollama doesn't require API key
            temperature=0.1,  # Low creativity for predictable testing
            request_timeout=30
        )
    else:
        pytest.skip("No gemma models available locally and could not pull")


@pytest.fixture
def mock_genome():
    """Create a test genome for consistent testing."""
    from allele.genome import ConversationalGenome
    return ConversationalGenome(
        genome_id="test_genome",
        traits={
            'empathy': 0.8,
            'engagement': 0.7,
            'technical_knowledge': 0.6,
            'creativity': 0.9,
            'conciseness': 0.5,
            'context_awareness': 0.8,
            'adaptability': 0.7,
            'personability': 0.8
        }
    )
