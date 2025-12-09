# Allele Runtime Testing Documentation

## Overview

Comprehensive runtime testing suite for the Allele genome-based conversational AI system, including unit, integration, performance, and stress tests.

## Test Structure

### Test Files

- **`tests/conftest.py`** - Shared pytest fixtures and configuration
- **`tests/test_utils.py`** - Testing utilities and helper functions
- **`tests/test_allele_genome.py`** - Original unit tests (enhanced)
- **`tests/test_genome_runtime.py`** - Runtime tests for genome operations
- **`tests/test_evolution_runtime.py`** - Runtime tests for evolution engine
- **`tests/test_kraken_lnn_runtime.py`** - Runtime tests for Kraken LNN
- **`tests/test_agent_runtime.py`** - Runtime tests for agent workflows
- **`tests/test_integration.py`** - End-to-end integration tests
- **`tests/test_performance.py`** - Performance benchmarks
- **`tests/test_stress.py`** - Stress tests and edge cases

## Running Tests

### All Tests

```bash
pytest tests/
```

### With Coverage

```bash
pytest tests/ --cov=allele --cov-report=html
```

### Specific Test Categories

```bash
# Unit runtime tests
pytest tests/test_genome_runtime.py tests/test_evolution_runtime.py

# Integration tests
pytest tests/test_integration.py

# Performance tests
pytest tests/test_performance.py --benchmark-only

# Stress tests
pytest tests/test_stress.py
```

### Parallel Execution

```bash
pytest tests/ -n auto  # Uses pytest-xdist
```

## Test Categories

### Unit Runtime Tests

Tests actual execution paths with real data:
- Genome creation and validation
- Trait access and modification
- Mutation operations
- Crossover operations
- Serialization/deserialization
- Gene expression synchronization

### Integration Tests

End-to-end workflows:
- Genome → Agent → Evolution → Kraken LNN
- Multi-component interactions
- Error propagation
- State management
- Conversation flows

### Performance Tests

Benchmarks for critical paths:
- Crossover latency (<5ms target)
- LNN processing latency (<10ms target)
- Memory usage profiling
- Throughput measurements
- Scalability analysis

### Stress Tests

Edge cases and limits:
- Large populations (1000+ genomes)
- Long evolution runs (100+ generations)
- Extended sequences (10K+ elements)
- Concurrent operations
- Memory buffer overflow
- Extreme trait values

## Test Coverage

Current coverage: **77.6%**

| Module | Coverage |
|--------|----------|
| genome.py | 85.1% |
| evolution.py | 75.6% |
| kraken_lnn.py | 68.9% |
| agent.py | 58.7% |
| types.py | 100% |
| exceptions.py | 85.4% |

## Fixtures

Common fixtures available in `conftest.py`:

- `sample_traits` - Sample trait dictionary
- `default_genome` - Genome with default traits
- `custom_genome` - Genome with custom traits
- `population_of_genomes` - Diverse population
- `evolution_engine` - Evolution engine instance
- `agent_config` - Agent configuration
- `kraken_lnn` - Kraken LNN instance
- `fitness_function` - Sample fitness function
- `performance_timer` - Performance timing fixture
- `memory_monitor` - Memory monitoring fixture

## Utilities

Testing utilities in `test_utils.py`:

- `generate_random_genome()` - Create random genome
- `generate_population()` - Create population
- `compare_genomes()` - Compare two genomes
- `calculate_population_diversity()` - Calculate diversity
- `assert_genome_valid()` - Validate genome
- `generate_test_sequence()` - Generate test sequences
- `measure_execution_time()` - Time function execution

## Performance Targets

- **Crossover**: <5ms (achieved: 2.3ms mean)
- **LNN Processing**: <10ms (achieved: 8.7ms mean)
- **Memory per Genome**: ~2KB (achieved: 2KB)
- **Test Execution**: <30s for full suite

## Continuous Integration

Tests are designed for CI/CD:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ --cov=allele --cov-report=xml
```

## Debugging

### Verbose Output

```bash
pytest tests/ -v
```

### Show Print Statements

```bash
pytest tests/ -s
```

### Run Specific Test

```bash
pytest tests/test_genome_runtime.py::TestGenomeRuntime::test_crossover_runtime_execution
```

### Debug on Failure

```bash
pytest tests/ --pdb
```

## Test Dependencies

Required packages (in `pyproject.toml`):

- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=4.0.0`
- `pytest-benchmark>=4.0.0`
- `pytest-timeout>=2.1.0`
- `pytest-xdist>=3.3.0`
- `memory-profiler>=0.61.0`

Install with:

```bash
pip install -e ".[dev]"
```

## Best Practices

1. **Use fixtures** for common test data
2. **Use utilities** for repeated operations
3. **Test edge cases** in stress tests
4. **Benchmark critical paths** in performance tests
5. **Verify state** after operations
6. **Use async/await** for async operations
7. **Set random seeds** for reproducibility

## Contributing

When adding new tests:

1. Follow existing test structure
2. Use appropriate fixtures
3. Add docstrings
4. Include edge cases
5. Update coverage targets
6. Run full test suite before committing

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)

