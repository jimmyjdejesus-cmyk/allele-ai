# Model Discovery and Size Detection

## ADDED Requirements

### REQ-DISCOVERY-001: Ollama Model Detection
The system MUST detect available Ollama models by querying the Ollama API or command-line interface.

#### Scenario: Detect available models
Given Ollama is running
And models are installed
When detect_ollama_models() is called
Then it queries Ollama for available models
And returns a list of model names
And handles connection errors gracefully

#### Scenario: Handle Ollama not running
Given Ollama is not running
When detect_ollama_models() is called
Then it logs an error message
And returns an empty list
And does not raise an exception

### REQ-DISCOVERY-002: Parameter Size Parsing
The system MUST parse model names to extract parameter counts and filter to 0.5b-3b range.

#### Scenario: Parse model name with explicit size
Given model name "qwen2.5:0.5b"
When parameter size is parsed
Then it extracts "0.5b"
And converts to 0.5 billion parameters
And includes in filtered list

#### Scenario: Parse various model name formats
Given model names:
- "gemma3:1b" → 1B parameters
- "llama3.2:1b" → 1B parameters
- "phi3:mini" → ~3.8B (excluded, >3B)
- "qwen2.5:0.5b" → 0.5B parameters
When parameter sizes are parsed
Then it correctly identifies sizes
And filters to 0.5b-3b range
And excludes models outside range

#### Scenario: Handle ambiguous model names
Given model name "tinyllama" (no explicit size)
When parameter size is parsed
Then it attempts to infer from model metadata
Or logs a warning
And excludes from auto-detection (requires manual specification)

### REQ-DISCOVERY-003: Size Range Filtering
The system MUST filter detected models to the 0.5b-3b parameter range.

#### Scenario: Filter models by size range
Given detected models:
- "qwen2.5:0.5b" (0.5B)
- "gemma3:1b" (1B)
- "llama3.2:1b" (1B)
- "phi3:mini" (3.8B)
- "llama3:8b" (8B)
When filtered to 0.5b-3b range
Then it includes: qwen2.5:0.5b, gemma3:1b, llama3.2:1b
And excludes: phi3:mini (3.8B > 3B), llama3:8b (8B > 3B)

#### Scenario: Handle edge cases
Given models at boundaries:
- "model:0.5b" (0.5B, included)
- "model:3b" (3B, included)
- "model:0.4b" (0.4B, excluded, <0.5B)
- "model:3.1b" (3.1B, excluded, >3B)
When filtered
Then boundary cases are handled correctly

### REQ-DISCOVERY-004: Manual Model Override
The system MUST support manual model specification to override auto-detection.

#### Scenario: Use manually specified models
Given --models flag with ["qwen2.5:0.5b", "gemma3:1b"]
When model discovery runs
Then it uses the manually specified models
And skips auto-detection
And validates models are available

#### Scenario: Combine auto-detection with manual override
Given auto-detection finds ["qwen2.5:0.5b"]
And --models flag specifies ["gemma3:1b"]
When model discovery runs
Then it includes both
And removes duplicates
And validates all models are available

### REQ-DISCOVERY-005: Model Availability Validation
The system MUST validate that detected or specified models are actually available in Ollama.

#### Scenario: Validate model availability
Given model name "qwen2.5:0.5b"
When availability is validated
Then it checks if model exists in Ollama
And returns True if available
And returns False if not available

#### Scenario: Handle unavailable model
Given model name "nonexistent:1b"
When availability is validated
Then it returns False
And logs a warning
And excludes from evaluation matrix

