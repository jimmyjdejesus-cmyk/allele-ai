# Expanded Multi-Model Benchmark Plan

## Model Testing Matrix

Based on your available models, here's a comprehensive testing matrix across different parameter sizes:

### Tier 1: Ultra-Small Models (≤1B)
- **tinyllama:latest** ✅ (already tested - creative_thinker +13.3%)
- **gemma:1b** - Google's 1B parameter model
- **qwen:0.5b** - Alibaba's small quantized model  
- **llama3.2:1b** - Meta's latest 1B model

### Tier 2: Small Models (1-4B)
- **gemma:2b** - Google's 2B parameter model
- **llama3.2:3b** - Meta's 3B model
- **qwen:0.6b** - Larger Qwen variant

### Tier 3: Medium Models (4-8B)
- **llama3.1:8b** - Meta's 8B model (instruction-tuned)
- **phi-2.7b** - Microsoft's enhanced Phi model

### Tier 4: Large Models (7B+ - Test for scaling effects)
- **mixtral-8x7b** - Mixture of Experts model (very large)

## Benchmark Types to Run

### 1. **Scaling Analysis**
Test how personality benefits change with model size:
- Run same personality suite across 0.5B → 1B → 2B → 3B → 8B models
- Measure diminishing returns of personality enhancement
- Identify optimal model size for personality optimization

### 2. **Architecture Comparison**
Compare personality effectiveness across different model families:
- **Meta Family**: llama3.2 (1B, 3B) vs llama3.1 (8B)
- **Google Family**: gemma (1B, 2B)
- **Alibaba Family**: qwen (0.5B, 0.6B)
- **Microsoft Family**: phi-2.7b
- **Mistral Family**: mixtral-8x7b

### 3. **Domain-Specific Personality Optimization**
Create specialized personality profiles for different model types:

#### **Code-Focused Models**
- **phi-2.7b**: Technical knowledge + conciseness + low creativity
- **codellama variants**: Programming-focused personalities

#### **Chat-Focused Models**
- **llama3.x family**: High engagement + personability + adaptability
- **gemma family**: Balanced personalities with high context awareness

#### **Instruction-Following Models**
- **llama3.1:8b**: Technical expertise + conciseness + context awareness

### 4. **Multi-Model Personality Transfer**
Test personality robustness:
- Train personality on small model, test on larger model
- Measure personality trait consistency across architectures
- Identify universal personality patterns

## Recommended Testing Strategy

### Phase 1: Baseline Scaling (2-3 hours)
```bash
# Test scaling across model sizes
python scripts/run_personality_benchmark.py --model gemma:1b --samples 15
python scripts/run_personality_benchmark.py --model llama3.2:3b --samples 15
python scripts/run_personality_benchmark.py --model llama3.1:8b --samples 15
```

### Phase 2: Architecture Comparison (1-2 hours)
```bash
# Compare different model families
python scripts/run_personality_benchmark.py --model qwen:0.6b --samples 15
python scripts/run_personality_benchmark.py --model phi-2.7b --samples 15
```

### Phase 3: Large Model Validation (30 minutes)
```bash
# Test largest model for scaling effects
python scripts/run_personality_benchmark.py --model mixtral-8x7b --samples 10
```

## Expected Research Insights

### Scaling Patterns
- **<1B models**: High personality impact (10-15% improvements)
- **1-3B models**: Moderate personality impact (5-10% improvements) 
- **3-8B models**: Lower personality impact (2-5% improvements)
- **8B+ models**: Minimal personality impact (diminishing returns)

### Architecture-Specific Findings
- **Meta models**: May prefer technical/analytical personalities
- **Google models**: Balanced across personality types
- **Microsoft models**: Strong technical bias
- **Alibaba models**: Flexible personality adaptation

### Universal Personalities
- **Technical Expert**: Likely works across all model sizes
- **Concise Analyst**: Probably universally beneficial
- **Creative Thinker**: May be most model-dependent

## Success Metrics

1. **Scaling Curve**: Quantify personality benefit vs model size
2. **Architecture Bias**: Identify model family preferences
3. **Universality**: Find personalities that work across all models
4. **Optimal Size**: Determine best model size for personality optimization

## Research Questions

1. **Do personality traits have diminishing returns with model size?**
2. **Are certain model architectures more personality-responsive?**
3. **Can we predict optimal personalities for new models?**
4. **What personality traits are universal vs model-specific?**

## Timeline
- **Phase 1**: 2-3 hours (baseline scaling)
- **Phase 2**: 1-2 hours (architecture comparison)
- **Phase 3**: 30 minutes (large model validation)
- **Analysis**: 1 hour (cross-model comparison)
- **Total**: ~5 hours for comprehensive study

This expanded approach will provide deep insights into how personality traits scale across different model architectures and sizes, potentially revealing fundamental principles of AI personality optimization.
