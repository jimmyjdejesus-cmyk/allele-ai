# API Reference

Phylogenic provides comprehensive REST APIs for programmatic access to all functionality. These APIs enable integration with external systems, monitoring, and management of Phylogenic deployments.

## OpenAPI Specifications

The APIs are formally defined using **OpenAPI 3.1 specifications** located in the [`docs/api/`](./api/) directory:

### Core API Categories

| API Category | Purpose | Specification File |
|-------------|---------|-------------------|
| **Agent Management** | Agent lifecycle, monitoring, configuration | [`agent.yaml`](./api/agent.yaml) |
| **Configuration** | Settings management, environments, profiles | [`config.yaml`](./api/config.yaml) |
| **Conversations** | Chat interactions, history, analytics | [`conversation.yaml`](./api/conversation.yaml) |
| **Genome Management** | Genome CRUD operations, analysis, versioning | [`genome.yaml`](./api/genome.yaml) |
| **Evolution Engine** | Genetic algorithm orchestration, monitoring | [`evolution.yaml`](./api/evolution.yaml) |
| **Shared Schemas** | Common data models, errors, pagination | [`schemas.yaml`](./api/schemas.yaml) |

## Getting Started

### Interactive Documentation

Generate live API documentation:

```bash
# Serve interactive API docs with Swagger UI
npx swagger-ui-watcher docs/api/agent.yaml

# Or use specialized OpenAPI tools
npm install -g @apidevtools/swagger-cli
swagger-cli validate docs/api/agent.yaml
swagger-cli bundle docs/api/agent.yaml --outfile bundled.json
```

### SDK Generation

Generate client SDKs in multiple languages:

```bash
# Python client
openapi-generator-cli generate \
  -i docs/api/agent.yaml \
  -g python \
  -o generated/python-client

# TypeScript client
openapi-generator-cli generate \
  -i docs/api/conversation.yaml \
  -g typescript-fetch \
  -o generated/typescript-client
```

## Architecture Overview

### Server Architecture

```
External Clients
     ↓
API Gateway/Load Balancer
     ↓
┌─────────────────────────────────────┐
│           API Services              │
├─────────────────────────────────────┤
│ Agent Service  │ Conversation Service │
│ Genome Service │ Evolution Service   │
│ Config Service │                     │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│      Phylogenic SDK Core Engine         │
│ - Genome Library                    │
│ - Evolution Engine                  │
│ - Kraken LNN                        │
│ - LLM Clients                       │
└─────────────────────────────────────┘
     ↓
Databases & External Services
```

### Design Principles

- **RESTful Design**: Resource-oriented URLs with consistent HTTP methods
- **JSON API**: All requests/responses use JSON format
- **Versioning**: URL path versioning (`/v1/`) with semantic versioning
- **Authentication**: Bearer token-based authentication
- **Pagination**: Consistent pagination for list endpoints
- **Error Handling**: Structured error responses with correlation IDs
- **Filtering/Sorting**: Standardized query parameters

## Key Concepts

### Authentication

All API endpoints require authentication via Bearer tokens:

```bash
# Set API token
export PHYLOGENIC_API_TOKEN="your_token_here"

# Example authenticated request
curl -H "Authorization: Bearer $PHYLOGENIC_API_TOKEN" \
     https://api.phylogenic.ai/v1/agents
```

### Core Data Models

The APIs work with Phylogenic's fundamental data structures:

#### Conversational Genome
8-trait personality encoding representing agent characteristics:

```json
{
  "genome_id": "customer_support_v2",
  "traits": {
    "empathy": 0.85,
    "technical_knowledge": 0.70,
    "creativity": 0.30,
    "conciseness": 0.80,
    "context_awareness": 0.90,
    "engagement": 0.88,
    "adaptability": 0.75,
    "personability": 0.92
  },
  "generation": 1
}
```

#### Agent Configuration
Complete agent setup including LLM provider and behavior settings:

```json
{
  "llm_provider": "openai",
  "model_name": "gpt-4-turbo-preview",
  "temperature": 0.7,
  "max_tokens": 2048,
  "streaming": true,
  "memory_enabled": true,
  "evolution_enabled": true,
  "kraken_enabled": true
}
```

### Error Handling

Consistent error response structure:

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": {
    "field": "temperature",
    "issue": "must be between 0 and 2"
  },
  "correlation_id": "req_12345",
  "timestamp": "2026-01-01T19:00:00Z"
}
```

### Pagination

All list endpoints support pagination:

```bash
# Get first 20 agents, page 1
curl "https://api.phylogenic.ai/v1/agents?page=1&page_size=20"

# Response includes pagination metadata
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_pages": 5,
    "total_items": 100,
    "has_next": true,
    "has_previous": false
  }
}
```

## Example Integrations

### Agent Management

```python
# Python client example
import requests

class PhylogenicClient:
    def __init__(self, base_url, api_token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def create_agent(self, genome_id, config):
        """Create and initialize a new agent"""
        payload = {
            "genome_id": genome_id,
            "config": config
        }
        response = requests.post(
            f"{self.base_url}/agents",
            json=payload,
            headers=self.headers
        )
        return response.json()

    def chat(self, agent_id, message):
        """Send message to agent"""
        payload = {"message": message}
        response = requests.post(
            f"{self.base_url}/agents/{agent_id}/chat",
            json=payload,
            headers=self.headers
        )
        return response.json()

# Usage
client = PhylogenicClient("https://api.phylogenic.ai/v1", "your_token")

agent = client.create_agent("support_agent_v1", {
    "llm_provider": "openai",
    "model_name": "gpt-4-turbo"
})

response = client.chat(agent["agent_id"], "Help me reset my password")
```

### Real-time Streaming Chat

```javascript
// WebSocket streaming for live chat
function startStreamingChat(agentId) {
    const wsUrl = `wss://api.phylogenic.ai/v1/agents/${agentId}/chat/stream`;
    const ws = new WebSocket(wsUrl, [], {
        headers: { "Authorization": `Bearer ${PHYLOGENIC_API_TOKEN}` }
    });

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'chunk') {
            displayChunk(data.content);
        } else if (data.type === 'finished') {
            showFinishedResponse(data.full_response);
        }
    };

    ws.onopen = () => {
        ws.send(JSON.stringify({
            message: "Explain machine learning",
            stream: true
        }));
    };

    return ws;
}
```

### Evolution Monitoring

```python
def monitor_evolution_run(client, run_id):
    """Monitor evolution progress in real-time"""
    while True:
        status = client.get_evolution_status(run_id)

        print(f"Generation {status['current_generation']}/{status['total_generations']}")
        print(f"Best Fitness: {status['best_fitness']:.4f}")
        print(f"Population Size: {status['population_size']}")

        if status["status"] == "completed":
            best_genome = client.get_genome(status["best_genome_id"])
            print(f"Evolution complete! Winner: {best_genome['genome_id']}")
            break

        time.sleep(5)  # Poll every 5 seconds
```

## Security

### Token Management

- **Bearer Authentication**: Required for all endpoints
- **Token Scopes**: Granular permissions (read, write, admin)
- **Expiration**: Tokens expire and must be renewed
- **Rotation**: Support for secure token rotation

### Data Protection

- **Encryption**: TLS 1.3 for all API communications
- **PII Masking**: Automatic detection and masking of personal data
- **Audit Logging**: Complete request/response logging for compliance

### Rate Limiting

- **Per-Token Limits**: Configurable requests-per-minute limits
- **Burst Handling**: Token bucket algorithm for traffic smoothing
- **Proper HTTP Status**: Returns 429 status for rate limit violations

## Testing & Validation

### Mock Server

Run a mock API server for testing:

```bash
# Using Prism
npm install -g @stoplight/prism
prism mock docs/api/agent.yaml --host 0.0.0.0 --port 4010

# Now test against mock server at http://localhost:4010
```

### Integration Testing

```python
# Using generated client
from phylogenic_api_client import ApiClient, Configuration

config = Configuration()
config.api_key = {"Authorization": PHYLOGENIC_API_TOKEN}
client = ApiClient(config)

# Test agent creation
try:
    agent = client.create_agent(genome_id="test", config={})
    assert agent.agent_id
    print("✓ Agent creation works")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
```

## Deployment

### Container Configuration

```dockerfile
FROM python:3.11-slim

# Copy API specifications for runtime validation
COPY docs/api/ /app/docs/api/

# Install dependencies
RUN pip install phylogenic-sdk fastapi uvicorn

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/v1/agents/health || exit 1

CMD ["uvicorn", "phylogenic.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

See the [`docs/api/README.md`](./api/README.md) for additional usage examples, tooling recommendations, and advanced integration patterns.
