# Real Integration Testing Guide

## Overview

Phylogenic's real integration testing framework validates actual LLM behavioral transformation and API connectivity with production-grade testing that uses real AI services, not mocks or stubs.

## Key Innovation: Zero-Mock Testing

**Traditional Testing**: Mock LLM responses
```python
# ❌ Fake testing
@pytest.fixture
def mock_openai_response():
    return {"choices": [{"message": {"content": "Fake response"}}]}

def test_fake_chat(mock_openai_response):
    # Tests nothing real
    pass
```

**Phylogenic Testing**: Real API calls with actual AI
```python
# ✅ Real testing
@pytest.mark.asyncio
async def test_ollama_cloud_real_chat():
    # Makes actual HTTPS calls to ollama.com
    agent = NLPAgent(genome, config)
    await agent.initialize()  # Real API authentication

    async for chunk in agent.chat("Hello"):
        collect_response(chunk)  # Real AI generation

    # Validates personality transformation actually works
```

## Environment Setup

### Ollama Cloud API Key
```bash
# Required for real cloud testing
export OLLAMA_API_KEY="<your-ollama-cloud-key>"
```

### OpenAI API Key (Optional)
```bash
export OPENAI_API_KEY="sk-your-openai-key"
```

### Test Environment Variables
```python
# Automatically set in test setup_method()
def setup_method(self):
    os.environ['OLLAMA_API_KEY'] = 'your_key_here'
```

## Running Real Integration Tests

### Basic Connectivity Test
```bash
# Test basic Ollama Cloud connection
pytest tests/test_llm_integration.py::TestLLMIntegration::test_ollama_cloud_real_initialization -xvs
```

### Full Conversational AI Test
```bash
# Test complete personality transformation
pytest tests/test_llm_integration.py::TestLLMIntegration::test_ollama_cloud_real_chat -xvs
```

### All Real Integration Tests
```bash
# Run all tests using real APIs
pytest tests/test_llm_integration.py -k "real" -v
```

## Test Architecture

### Multi-Provider Validation Strategy

```python
# Try multiple models sequentially until one works
models_to_try = ["llama2", "llama2:7b", "mistral", "codellama"]

for model_name in models_to_try:
    try:
        config = AgentConfig(llm_provider="ollama", model_name=model_name)
        agent = NLPAgent(genome, config)
        await agent.initialize()  # Real API call
        # Success - use this model for testing
        break
    except Exception:
        continue  # Try next model
```

### Behavioral Validation Framework

```python
# Test that genome actually changes LLM behavior
genome_empathetic = create_genome(empathy=0.9, personability=0.1)
genome_technical = create_genome(empathy=0.1, technical_knowledge=0.9)

# Response comparison validation
response_empathetic = await chat_with_genome(genome_empathetic, "I feel sad")
response_technical = await chat_with_genome(genome_technical, "Code not working")

assert "comforting" in response_empathetic.lower()
assert "debug" in response_technical.lower()
```

## Test Categories

### 🚀 Production-Ready Integration Tests

#### Real API Connectivity
- ✅ Actual HTTPS calls to cloud providers
- ✅ Authentication validation
- ✅ SSL/TLS verification
- ✅ Error handling for network issues

#### Behavioral Transformation Validation
- ✅ Genome traits modify system prompts dynamically
- ✅ LLM responses reflect personality injection
- ✅ Context management preserves coherence
- ✅ Kraken LNN enhances conversation flow

#### Multi-Provider Compatibility
- ✅ OpenAI GPT models via API
- ✅ Ollama local models (localhost:11434)
- ✅ Ollama cloud models (ollama.com)
- ✅ Authentication for cloud services

### 🔬 Research-Grade Validation

#### Personality Differentiation
```python
# Test that different genomes produce measurably different responses
responses = []
for genome in [creative_genome, analytical_genome, empathetic_genome]:
    response = await test_standard_question(genome, "Should I quit my job?")
    responses.append(response)

# Validate behavioral diversity
assert not all_same(responses)  # Each genome produces unique response
assert personality_metrics_match_expected(responses, genomes)
```

#### Context Coherence Testing
```python
# Test memory retention over multi-turn conversations
conversation = [
    "Hello, I'm learning Python",
    "How do I create a function?",
    "Great! Now what about classes?"
]

for message in conversation:
    response = await agent.chat(message)

# Validate context awareness
assert "function" referenced in response_to_class_question
assert "Python" context maintained throughout
```

## Performance Testing

### Real API Latency Validation
```python
start_time = time.time()
response = await agent.chat(user_message)  # Real LLM call
latency = time.time() - start_time

assert latency < 10.0  # Sub-10 second response requirement
assert len(response) > 10  # Substantial response
```

### Rate Limiting Compliance
```python
# Test automatic rate limiting
for i in range(100):
    await agent.chat(f"Test message {i}")
    # Should not hit API rate limits

# Validate cost tracking
costs = await agent.get_metrics()['llm_cost']
assert costs > 0.0  # Actual API calls made
```

## Error Scenarios & Recovery

### Network Failure Testing
```python
# Test offline -> online recovery
with patch_network_down():
    with pytest.raises(LLMTimeoutError):
        await agent.chat("Test")

# Test automatic retry
with patch_network_intermittent():
    response = await agent.chat("Test")
    assert response is not None  # Recovery worked
```

### API Quota Management
```python
# Test low quota handling
with patch_quota_exceeded():
    # Should gracefully degrade or fallback
    response = await agent.chat("Test")
    assert "FALLBACK MODE" in response
```

## Quality Assurance Metrics

### Test Coverage Validation
```bash
pytest --cov=phylogenic --cov-report=term-missing tests/test_llm_integration.py
```

### Behavioral Consistency Scoring
```python
def test_personality_consistency():
    # Send same message to same genome multiple times
    responses = []
    for _ in range(5):
        response = await agent.chat("Tell me about yourself")
        responses.append(response)

    # Calculate personality consistency score
    consistency = calculate_trait_adherence(responses, agent.genome)
    assert consistency > 0.8  # 80% personality consistency
```

## Cost-Effective Testing Strategy

### Graduated Testing Approach
1. **Unit Tests** - No API calls, fast feedback
2. **Integration Tests** - Minimal API calls, basic connectivity
3. **Behavioral Tests** - Full conversations, personality validation
4. **Stress Tests** - High volume, rate limiting validation

### Cost Management
```python
# Limit expensive tests in CI/CD
if os.getenv('CI') and 'expensive' in pytest.marks:
    pytest.skip("Skipping expensive real API tests in CI")

# Run full suite only on main branch or tagged releases
```

## Troubleshooting Real Integration Tests

### Common Issues

**API Key Missing**
```bash
export OLLAMA_API_KEY="your-key-here"
# Or check .env file
```

**Network Timeout**
```python
config = AgentConfig(request_timeout=120)  # Increase timeout
```

**Model Not Available**
```python
# Test different models
config = AgentConfig(model_name="llama2")  # Try simpler model
```

**SSL Errors**
```python
# Check proxies or corporate networks
# May need to disable SSL verification in test environments
```

## Production Deployment Validation

### Pre-Deployment Checklist
- [ ] Ollama Cloud API key configured
- [ ] OpenAI API key optional (fallback mode available)
- [ ] Network connectivity to ollama.com verified
- [ ] SSL certificates valid
- [ ] Rate limiting within API limits

### Post-Deployment Monitoring
```python
# Check real metrics
metrics = await agent.get_metrics()
assert metrics['performance']['error_rate'] < 0.05  # <5% errors
assert metrics['llm']['total_tokens_used'] > 0     # API actually called
```

## Extending Real Integration Testing

### Adding New Providers
```python
# Template for new provider integration
class NewProviderClient(LLMClient):
    async def initialize(self):
        # Real API authentication
        pass

    async def chat_completion(self, messages, stream=True):
        # Real API calls
        pass
```

### Custom Personality Tests
```python
def test_custom_personality(genome_config):
    """Parameterized test for different personality combinations"""
    genome = ConversationalGenome(
        genome_id=f"custom_{genome_config['id']}",
        traits=genome_config['traits']
    )
    agent = NLPAgent(genome, config)
    # Real validation tests...
```

## Summary

Phylogenic's real integration testing framework provides:

- **✅ Zero-Mock Validation** - Actual AI service interaction
- **✅ Behavioral Transformation Proof** - Genome changes LLM responses
- **✅ Production Readiness** - Real API connectivity testing
- **✅ Multi-Provider Support** - OpenAI, Ollama local/cloud
- **✅ Cost-Effective** - Targeted real API usage
- **✅ Error Recovery Testing** - Network failures, quotas, timeouts

This represents elite-level software testing - validating actual AI behavioral transformation rather than mocked interfaces.

**Real Integration Testing = Real AI Validation** 🚀
