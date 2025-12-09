from allele import settings
from allele.genome import ConversationalGenome
from allele.kraken_lnn import KrakenLNN


def test_genome_defaults_from_settings():
    # No traits supplied -> should use central defaults
    genome = ConversationalGenome.from_settings("g_default")
    assert genome.genome_id == "g_default"
    assert genome.traits == settings.default_traits


def test_kraken_from_settings():
    kraken = KrakenLNN.from_settings()
    assert kraken.reservoir_size == settings.kraken.reservoir_size
    assert kraken.connectivity == settings.kraken.connectivity
    assert kraken.temporal_memory.buffer_size == settings.kraken.memory_buffer_size
    assert kraken.dynamics.temperature == settings.liquid_dynamics.temperature
