#!/bin/bash
# Usage: ./scripts/quick_benchmark.sh [model] [samples]

MODEL=${1:-gemma3:1b}
SAMPLES=${2:-20}

echo "🚀 Starting Quick Benchmark"
echo "Model: $MODEL"
echo "Samples: $SAMPLES"
echo "-----------------------------------"

# Ensure script is executable
chmod +x scripts/run_lm_eval_mass.py

# Run the python runner in quick mode
python3 scripts/run_lm_eval_mass.py \
  --mode quick \
  --models "$MODEL" \
  --limit "$SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ Benchmark completed successfully"
    
    # Run analysis
    echo "📊 Generating report..."
    python3 scripts/analyze_lm_eval_results.py
else
    echo "❌ Benchmark failed"
    exit 1
fi

