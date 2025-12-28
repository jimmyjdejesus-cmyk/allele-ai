# Multi-Model Benchmark Plan

## Overview
Expand the personality benchmark framework to test across multiple small models (<1B parameters) to validate the universality of genetic personality traits across different model architectures.

## Models to Test

### Tier 1: Proven Models (Run First)
- **tinyllama:latest** ✅ (already completed)
- **gemma:1b** - Google's 1B parameter model
- **qwen:0.5b** - Alibaba's small quantized model  
- **mistral:1b** - Mistral's 1B parameter model

### Tier 2: Experimental Models
- **phi:1b** - Microsoft's Phi-1
- **codellama:1b** - Code-focused variant
- **llama3:1b** - Meta's latest small model

### Tier 3: Emerging Models (if time allows)
- **qwen:1.5b** - Larger Qwen variant
- **gemma2:1b** - Google's latest 1B model
- **openchat:1b** - Open-source chat model

## Benchmark Types

### 1. **Personality Consistency Test**
- Run full personality suite on each model
- Compare which personalities work best for each architecture
- Identify universal vs model-specific personality traits

### 2. **Cross-Model Personality Transfer**
- Train personality on one model, test transfer to another
- Measure personality trait robustness across architectures

### 3. **Performance Scaling Analysis**
- Test same personality traits across model sizes (0.5b, 1b, 1.5b)
- Analyze scaling effects of personality enhancement

### 4. **Domain-Specific Personalities**
- **Code Expert**: Technical knowledge + conciseness for programming
- **Creative Writer**: High creativity + engagement for content generation
- **Analyst**: High technical knowledge + context awareness for data analysis

## Recommended Command Structure

```bash
# Run personality benchmarks on multiple models
python scripts/run_personality_benchmark.py --model gemma:1b --samples 20
python scripts/run_personality_benchmark.py --model qwen:0.5b --samples 20
python scripts/run_personality_benchmark.py --model mistral:1b --samples 20
python scripts/run_personality_benchmark.py --model phi:1b --samples 20

# Compare results across models
python scripts/analyze_cross_model_results.py --model-list gemma:1b,qwen:0.5b,mistral:1b
```

## Expected Insights

### Universal Personalities
- **Technical Expert** likely works across all models
- **Concise Analyst** probably universally beneficial
- **Creative Thinker** may vary more by model

### Model-Specific Patterns
- **Code-focused models** (CodeLlama, Phi) may prefer technical personalities
- **Chat-focused models** may prefer engaging personalities
- **Instruction-tuned models** may prefer analytical personalities

### Performance Scaling
- Larger models may show smaller personality improvements (diminishing returns)
- Smaller models may benefit more from personality enhancement
- Personality traits may be more critical for smaller models

## Timeline Estimate

- **Tier 1 Models** (4 models): 2-3 hours total runtime
- **Analysis & Comparison**: 1 hour
- **Documentation**: 30 minutes
- **Total**: ~4 hours for comprehensive multi-model study

## Success Metrics

1. **Consistency**: Same personality traits show similar patterns across models
2. **Universality**: At least 1 personality works well across all models tested
3. **Novelty**: Discover model-specific personality preferences
4. **Scaling**: Quantify how personality benefits change with model size

## Next Steps

1. **Start with Tier 1 models** - run personality benchmarks on gemma:1b, qwen:0.5b, mistral:1b
2. **Compare results** - identify universal vs model-specific patterns  
3. **Expand based on findings** - test Tier 2 models based on initial results
4. **Create cross-model analysis** - build comprehensive comparison matrix

This multi-model approach will validate the robustness of our personality framework and potentially discover model-specific optimization strategies.
