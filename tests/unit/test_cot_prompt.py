"""Tests for Chain of Thought prompt functionality."""

import asyncio
from types import SimpleNamespace

from phylogenic.benchmark.utils import build_cot_prompt, build_system_prompt
from scripts.run_personality_benchmark import GenomeModel


class FakeClient:
    """Mock Ollama client for testing."""
    
    def __init__(self, response: str):
        self.response = response
        self.last_messages = None
    
    async def chat_completion(self, messages, stream=False):
        self.last_messages = messages
        yield self.response
    
    async def close(self):
        pass


async def _run_generate(model, prompt):
    """Helper to run async generate."""
    return await model.generate(prompt)


def test_build_cot_prompt_basic():
    """Test basic COT prompt wrapping."""
    prompt = "What is 2+2?"
    result = build_cot_prompt(prompt)
    assert "Let's think step by step:" in result
    assert prompt in result
    assert result.endswith("Let's think step by step:")


def test_build_cot_prompt_empty():
    """Test COT prompt with empty input."""
    result = build_cot_prompt("")
    assert result == "Let's think step by step:"


def test_build_cot_prompt_multiline():
    """Test COT prompt with multiline input."""
    prompt = "Question 1: What is 2+2?\nQuestion 2: What is 3+3?"
    result = build_cot_prompt(prompt)
    assert prompt in result
    assert "\n\nLet's think step by step:" in result


def test_genomemodel_cot_mode_without_genome():
    """Test GenomeModel with COT mode but no genome."""
    fake = FakeClient("Answer: 4")
    model = GenomeModel(fake, None, cot_mode=True)
    
    resp = asyncio.run(_run_generate(model, "What is 2+2?"))
    assert resp == "Answer: 4"
    assert fake.last_messages is not None
    assert len(fake.last_messages) == 1  # No system prompt
    assert "Let's think step by step:" in fake.last_messages[0]["content"]


def test_genomemodel_cot_mode_with_genome():
    """Test GenomeModel with both COT mode and genome traits."""
    traits = {"technical_knowledge": 0.9}
    genome = SimpleNamespace(traits=traits)
    fake = FakeClient("Answer: 4")
    model = GenomeModel(fake, genome, cot_mode=True)
    
    resp = asyncio.run(_run_generate(model, "What is 2+2?"))
    assert resp == "Answer: 4"
    assert fake.last_messages is not None
    assert len(fake.last_messages) == 2  # System + user
    assert fake.last_messages[0]["role"] == "system"
    assert fake.last_messages[1]["role"] == "user"
    assert "Let's think step by step:" in fake.last_messages[1]["content"]
    assert build_system_prompt(traits) in fake.last_messages[0]["content"]


def test_genomemodel_cot_mode_disabled():
    """Test GenomeModel with COT mode disabled (default)."""
    fake = FakeClient("Answer: 4")
    model = GenomeModel(fake, None, cot_mode=False)
    
    resp = asyncio.run(_run_generate(model, "What is 2+2?"))
    assert resp == "Answer: 4"
    assert "Let's think step by step:" not in fake.last_messages[0]["content"]


def test_genomemodel_backward_compatibility():
    """Test that GenomeModel without cot_mode parameter still works."""
    fake = FakeClient("Answer: 4")
    model = GenomeModel(fake, None)  # cot_mode defaults to False
    
    resp = asyncio.run(_run_generate(model, "What is 2+2?"))
    assert resp == "Answer: 4"
    assert "Let's think step by step:" not in fake.last_messages[0]["content"]

