# Comprehensive Guide to LM-Eval Benchmarking

This guide details how to use the new `lm-eval` (Language Model Evaluation Harness) infrastructure for robust, standardized LLM benchmarking on M1/M2 Macs.

## 🚀 Quick Start

### 1. Installation
Ensure you have the required dependencies:
```bash
pip install "lm-eval[all]>=0.4.0" pandas tabulate
```

### 2. Run a Quick Test
Validate your setup with a 10-minute quick benchmark:
```bash
./scripts/quick_benchmark.sh gemma3:1b 20
```

### 3. Run Standard Benchmarks
Run a comprehensive suite (MMLU, HellaSwag, GSM8K, ARC, TruthfulQA):
```bash
python scripts/run_lm_eval_mass.py --mode standard --models gemma3:1b qwen2.5:0.5b
```

---

## 🛠️ Architecture

The benchmarking pipeline consists of three main components:

1. **Runner (`scripts/run_lm_eval_mass.py`)**: 
   - Orchestrates `lm-eval` execution
   - Manages Ollama models (auto-pulls if missing)
   - Handles batching and checkpoints

2. **Analyzer (`scripts/analyze_lm_eval_results.py`)**:
   - Parses raw JSON results
   - Generates comparison tables
   - Produces Markdown reports

3. **Updater (`scripts/update_readme_benchmarks.py`)**:
   - Injects latest results into `README.md` automatically

---

## 📊 Benchmark Modes

| Mode | Tasks | Est. Time (per model) | Use Case |
|------|-------|-----------------------|----------|
| `quick` | MMLU, HellaSwag, GSM8K (Limited samples) | ~5-10 mins | Smoke testing, verifying setup |
| `standard` | MMLU, HellaSwag, GSM8K, ARC-Easy, TruthfulQA | ~1-2 hours | Routine evaluation, A/B testing |
| `comprehensive` | Above + ARC-Challenge, Winogrande, PIQA, BoolQ, SiQA | ~8-12 hours | Final model validation, publication |

---

## 💻 Hardware Optimization (M1/M2 Mac)

The pipeline is pre-configured for Apple Silicon:

- **Backend**: Uses `local-completions` via Ollama API
- **Batch Size**: Forced to `1` to prevent memory overflows on 8GB/16GB unified memory
- **Models**: Defaults to small, high-performance models:
  - `gemma3:1b`: Excellent reasoning/size ratio
  - `qwen2.5:0.5b`: Extremely fast, good for quick checks
  - `llama3.2:1b`: Balanced general purpose
  - `phi3:mini`: (3.8B) High capability if memory allows

---

## 📈 Interpreting Results

Results are saved to `benchmark_results/lm_eval/`:

- **MMLU**: Knowledge breadth. <25% is random chance. >40% is good for 1B models.
- **HellaSwag**: Commonsense. >60% is decent for small models.
- **GSM8K**: Math reasoning. Often low (<10%) for 1B models without fine-tuning.
- **TruthfulQA**: Safety/Hallucination. Higher is better (more truthful).

### Example Output Table

| Model | Average | MMLU | HellaSwag | GSM8K |
|-------|---------|------|-----------|-------|
| gemma3:1b | 45.2 | 42.1 | 65.4 | 12.5 |
| qwen2.5:0.5b | 38.7 | 35.2 | 58.9 | 8.4 |

---

## 🔧 Troubleshooting

**"Ollama is not running"**
- Run `ollama serve` in a separate terminal.

**"Connection refused"**
- Ensure Ollama is on port 11434 (default).

**"Out of Memory (OOM)"**
- Don't run `phi3:mini` or larger models alongside other heavy apps.
- Close browser tabs/IDEs during comprehensive runs.

**"Results show 0.0"**
- Check if the model output format matches what `lm-eval` expects.
- Ensure `num_fewshot` is 0 (zero-shot) for consistent baselines.

---

## 📝 Adding New Benchmarks

To add a new task (e.g., `codex`), update `scripts/run_lm_eval_mass.py`:

```python
BENCHMARK_TASKS = {
    "custom": ["codex", "humaneval"],
    # ...
}
```

Then run:
```bash
python scripts/run_lm_eval_mass.py --mode custom
```

