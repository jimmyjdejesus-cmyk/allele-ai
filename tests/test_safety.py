import pytest

from allele.agent import AgentConfig
from allele.evolution import EvolutionConfig, EvolutionEngine
from allele.kraken_lnn import LiquidStateMachine


def test_large_reservoir_rejected():
    with pytest.raises(ValueError):
        LiquidStateMachine(reservoir_size=100000)


def test_large_population_rejected():
    cfg = EvolutionConfig(population_size=100000)
    with pytest.raises(ValueError):
        EvolutionEngine(cfg)


def test_large_conversation_memory_rejected():
    with pytest.raises(ValueError):
        AgentConfig(conversation_memory=10_000)


def test_large_context_window_rejected():
    with pytest.raises(ValueError):
        AgentConfig(context_window=1000)
