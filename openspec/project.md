# Phylogenic-AI-Agents Project Specification

## Overview

Phylogenic-AI-Agents is a framework for genome-based conversational AI agents with evolutionary optimization and liquid neural networks. The system enables reproducible, evolvable, and explainable agent personalities through genetic encoding rather than static prompt engineering.

## Core Capabilities

### Benchmarking System
- Personality-based A/B testing (`run_personality_benchmark.py`)
- LM-Eval integration for standardized benchmarks (`run_lm_eval_mass.py`)
- Custom benchmark implementations (MMLU, GSM8K, HellaSwag, etc.)
- Results aggregation and analysis

### Genome System
- 8-trait personality encoding (empathy, technical_knowledge, creativity, conciseness, context_awareness, adaptability, engagement, personability)
- System prompt generation from genome traits
- Personality archetypes (technical_expert, creative_thinker, concise_analyst, balanced, high_context)

### LLM Integration
- Multi-provider support (OpenAI, Anthropic, Ollama)
- Local model execution via Ollama
- Streaming responses
- Conversation memory

## Current Limitations

- No comprehensive multi-model evaluation matrix
- No Chain of Thought (COT) prompt support
- No automated model discovery for parameter range filtering
- Manual documentation updates for benchmark results
- Limited cross-model personality comparison capabilities

