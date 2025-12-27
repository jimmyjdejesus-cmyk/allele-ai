#!/usr/bin/env python3
"""
Multi-Personality A/B Benchmark Runner

Tests how different genome personality configurations affect LLM performance.

Usage:
    python scripts/run_personality_benchmark.py --model tinyllama:latest --samples 30
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from phylogenic.benchmark.utils import check_answer, build_system_prompt
from src.phylogenic.genome import ConversationalGenome
from src.phylogenic.llm_client import LLMConfig
from src.phylogenic.llm_ollama import OllamaClient

# Define personality archetypes to test
PERSONALITY_ARCHETYPES = {
    "baseline": None,  # No genome, raw model
    "technical_expert": {
        "empathy": 0.2,
        "technical_knowledge": 0.99,
        "creativity": 0.3,
        "conciseness": 0.95,
        "context_awareness": 0.9,
        "adaptability": 0.5,
        "engagement": 0.2,
        "personability": 0.2
    },
    "creative_thinker": {
        "empathy": 0.7,
        "technical_knowledge": 0.6,
        "creativity": 0.99,
        "conciseness": 0.4,
        "context_awareness": 0.7,
        "adaptability": 0.9,
        "engagement": 0.8,
        "personability": 0.7
    },
    "concise_analyst": {
        "empathy": 0.3,
        "technical_knowledge": 0.85,
        "creativity": 0.4,
        "conciseness": 0.99,
        "context_awareness": 0.8,
        "adaptability": 0.6,
        "engagement": 0.3,
        "personability": 0.3
    },
    "balanced": {
        "empathy": 0.5,
        "technical_knowledge": 0.7,
        "creativity": 0.5,
        "conciseness": 0.6,
        "context_awareness": 0.7,
        "adaptability": 0.7,
        "engagement": 0.5,
        "personability": 0.5
    },
    "high_context": {
        "empathy": 0.6,
        "technical_knowledge": 0.8,
        "creativity": 0.5,
        "conciseness": 0.7,
        "context_awareness": 0.99,
        "adaptability": 0.8,
        "engagement": 0.5,
        "personability": 0.5
    }
}


@dataclass
class PersonalityResult:
    """Results for a single personality archetype."""
    personality_name: str
    traits: Dict[str, float]
    mmlu_score: float
    gsm8k_score: float
    reasoning_score: float
    average_score: float
    total_time: float


class GenomeModel:
    """Model with optional genome enhancement."""

    def __init__(self, client: OllamaClient, genome: ConversationalGenome = None):
        self.client = client
        self.genome = genome

    def _build_system_prompt(self) -> str:
        """Delegate prompt building to shared utility to avoid duplication."""
        if self.genome is None:
            return ""
        return build_system_prompt(self.genome.traits)

    async def generate(self, prompt: str) -> str:
        system_prompt = self._build_system_prompt()

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = ""
        async for chunk in self.client.chat_completion(messages, stream=False):
            response += chunk
        return response.strip()


# Benchmark samples
MMLU_SAMPLES = [
    {"prompt": "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:", "expected": "C"},
    {"prompt": "Question: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer:", "expected": "B"},
    {"prompt": "Question: What is the chemical symbol for gold?\nA. Go\nB. Gd\nC. Au\nD. Ag\nAnswer:", "expected": "C"},
    {"prompt": "Question: Who wrote 'Romeo and Juliet'?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain\nAnswer:", "expected": "B"},
    {"prompt": "Question: What is the largest organ in the human body?\nA. Heart\nB. Brain\nC. Liver\nD. Skin\nAnswer:", "expected": "D"},
    {"prompt": "Question: What is the speed of light in vacuum?\nA. 300,000 km/s\nB. 150,000 km/s\nC. 450,000 km/s\nD. 600,000 km/s\nAnswer:", "expected": "A"},
    {"prompt": "Question: Which gas do plants absorb?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nAnswer:", "expected": "C"},
    {"prompt": "Question: What is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:", "expected": "C"},
    {"prompt": "Question: In what year did World War II end?\nA. 1943\nB. 1944\nC. 1945\nD. 1946\nAnswer:", "expected": "C"},
    {"prompt": "Question: What is the main function of red blood cells?\nA. Fight infection\nB. Carry oxygen\nC. Clot blood\nD. Digest food\nAnswer:", "expected": "B"},
    {"prompt": "Question: Which element has atomic number 1?\nA. Helium\nB. Hydrogen\nC. Lithium\nD. Carbon\nAnswer:", "expected": "B"},
    {"prompt": "Question: What is the derivative of x squared?\nA. x\nB. 2x\nC. x squared\nD. 2\nAnswer:", "expected": "B"},
    {"prompt": "Question: Which continent is the Sahara Desert on?\nA. Asia\nB. Australia\nC. Africa\nD. South America\nAnswer:", "expected": "C"},
    {"prompt": "Question: What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi\nAnswer:", "expected": "C"},
    {"prompt": "Question: Who developed the theory of relativity?\nA. Newton\nB. Einstein\nC. Bohr\nD. Planck\nAnswer:", "expected": "B"},
    {"prompt": "Question: What is the boiling point of water at sea level?\nA. 90C\nB. 100C\nC. 110C\nD. 120C\nAnswer:", "expected": "B"},
    {"prompt": "Question: Which is the longest river in the world?\nA. Amazon\nB. Nile\nC. Yangtze\nD. Mississippi\nAnswer:", "expected": "B"},
    {"prompt": "Question: What is the chemical formula for water?\nA. CO2\nB. H2O\nC. NaCl\nD. O2\nAnswer:", "expected": "B"},
    {"prompt": "Question: Who painted the Mona Lisa?\nA. Michelangelo\nB. Raphael\nC. Da Vinci\nD. Picasso\nAnswer:", "expected": "C"},
    {"prompt": "Question: What is the largest planet in our solar system?\nA. Earth\nB. Saturn\nC. Jupiter\nD. Neptune\nAnswer:", "expected": "C"},
]

GSM8K_SAMPLES = [
    {"prompt": "If 5 apples cost $2, how much do 15 apples cost? Answer with just the number:", "expected": "6"},
    {"prompt": "A train travels 120 miles in 2 hours. What is its speed in mph? Answer:", "expected": "60"},
    {"prompt": "John has 24 candies, gives away 1/3. How many left? Answer:", "expected": "16"},
    {"prompt": "Rectangle length 8, width 5. What is the area? Answer:", "expected": "40"},
    {"prompt": "If 3x + 7 = 22, what is x? Answer:", "expected": "5"},
    {"prompt": "Book costs $15, 20% discount. New price? Answer:", "expected": "12"},
    {"prompt": "What is 15% of 200? Answer:", "expected": "30"},
    {"prompt": "Car uses 5 gallons for 150 miles. Miles per gallon? Answer:", "expected": "30"},
    {"prompt": "Sum of integers 1 to 10? Answer:", "expected": "55"},
    {"prompt": "12 times 8 equals? Answer:", "expected": "96"},
]

REASONING_SAMPLES = [
    {"prompt": "The man put milk in the fridge because:\nA. Empty\nB. Needed cold\nC. Hungry\nD. Door open\nAnswer:", "expected": "B"},
    {"prompt": "After rain stopped, streets were:\nA. Dry\nB. Wet\nC. Hot\nD. Dark\nAnswer:", "expected": "B"},
    {"prompt": "Bird flew south for winter because:\nA. Bored\nB. Tourism\nC. Cold weather\nD. Lost\nAnswer:", "expected": "C"},
    {"prompt": "She brought umbrella because:\nA. Sunny\nB. Exercise\nC. Rain forecast\nD. Cold\nAnswer:", "expected": "C"},
    {"prompt": "Baby crying, mother:\nA. Slept\nB. Left\nC. Comforted\nD. TV\nAnswer:", "expected": "C"},
    {"prompt": "Ice cream melted because:\nA. Freezer\nB. Hot day\nC. Eaten\nD. Chocolate\nAnswer:", "expected": "B"},
    {"prompt": "Studied all night because:\nA. Nothing to do\nB. Exam tomorrow\nC. Bored\nD. Rain\nAnswer:", "expected": "B"},
    {"prompt": "Plant died because:\nA. Too much sun\nB. No water\nC. Too tall\nD. Green\nAnswer:", "expected": "B"},
    {"prompt": "Wore sunglasses because:\nA. Dark\nB. Rain\nC. Bright sun\nD. Tired\nAnswer:", "expected": "C"},
    {"prompt": "Car won't start because:\nA. New\nB. Dead battery\nC. Parked\nD. Red\nAnswer:", "expected": "B"},
]


# Use the shared `check_answer` utility from `src.benchmark.utils` to avoid
# duplicated and potentially-buggy implementations across scripts.


async def evaluate_samples(model: GenomeModel, samples: List[Dict]) -> tuple[float, int]:
    """Evaluate model on samples, return (score%, correct_count)."""
    correct = 0
    for sample in samples:
        try:
            response = await model.generate(sample["prompt"])
            if check_answer(response, sample["expected"]):
                correct += 1
        except Exception as e:
            print(f"    Error: {e}")
    score = (correct / len(samples)) * 100 if samples else 0
    return score, correct


async def run_personality_benchmark(model_name: str, max_samples: int = 30):
    """Run benchmarks for all personality archetypes."""

    print("=" * 70)
    print("PHYLOGENIC PERSONALITY A/B BENCHMARK")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Samples per benchmark: {max_samples}")
    print(f"Personalities to test: {list(PERSONALITY_ARCHETYPES.keys())}")
    print()

    # Initialize client
    config = LLMConfig(
        provider="ollama",
        model=model_name,
        temperature=0.1,
        max_tokens=256,
        timeout=120
    )

    client = OllamaClient(config)

    try:
        await client.initialize()
        print("[OK] Connected to Ollama")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # Prepare samples
    mmlu = MMLU_SAMPLES[:max_samples]
    gsm8k = GSM8K_SAMPLES[:max(max_samples // 3, 3)]
    reasoning = REASONING_SAMPLES[:max(max_samples // 3, 3)]

    results: List[PersonalityResult] = []

    # Test each personality
    for name, traits in PERSONALITY_ARCHETYPES.items():
        print(f"\n{'-' * 60}")
        print(f"Testing: {name.upper()}")
        if traits:
            key_traits = [f"{k}={v:.1f}" for k, v in traits.items() if v > 0.7]
            print(f"High traits: {', '.join(key_traits) if key_traits else 'balanced'}")
        print("-" * 60)

        # Create model with genome
        if traits:
            genome = ConversationalGenome(genome_id=f"personality_{name}", traits=traits)
            model = GenomeModel(client, genome)
        else:
            model = GenomeModel(client, None)

        start = time.time()

        # Run benchmarks
        print(f"  MMLU ({len(mmlu)} samples)...", end=" ", flush=True)
        mmlu_score, mmlu_correct = await evaluate_samples(model, mmlu)
        print(f"{mmlu_score:.1f}% ({mmlu_correct}/{len(mmlu)})")

        print(f"  GSM8K ({len(gsm8k)} samples)...", end=" ", flush=True)
        gsm8k_score, gsm8k_correct = await evaluate_samples(model, gsm8k)
        print(f"{gsm8k_score:.1f}% ({gsm8k_correct}/{len(gsm8k)})")

        print(f"  Reasoning ({len(reasoning)} samples)...", end=" ", flush=True)
        reasoning_score, reasoning_correct = await evaluate_samples(model, reasoning)
        print(f"{reasoning_score:.1f}% ({reasoning_correct}/{len(reasoning)})")

        elapsed = time.time() - start
        avg = (mmlu_score + gsm8k_score + reasoning_score) / 3

        print(f"  AVERAGE: {avg:.1f}% (in {elapsed:.1f}s)")

        results.append(PersonalityResult(
            personality_name=name,
            traits=traits or {},
            mmlu_score=mmlu_score,
            gsm8k_score=gsm8k_score,
            reasoning_score=reasoning_score,
            average_score=avg,
            total_time=elapsed
        ))

    await client.close()

    # Print summary matrix
    print("\n" + "=" * 70)
    print("RESULTS MATRIX")
    print("=" * 70)

    # Find baseline for comparison
    baseline_avg = next((r.average_score for r in results if r.personality_name == "baseline"), 0)

    print("\n| Personality | MMLU | GSM8K | Reasoning | AVG | vs Baseline |")
    print("|-------------|------|-------|-----------|-----|-------------|")

    for r in results:
        delta = r.average_score - baseline_avg
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        status = "[+]" if delta > 2 else "[-]" if delta < -2 else "[=]"
        print(f"| {r.personality_name:11} | {r.mmlu_score:4.1f}% | {r.gsm8k_score:5.1f}% | {r.reasoning_score:9.1f}% | {r.average_score:3.1f}% | {delta_str:6} {status} |")

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    results_data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "samples": {"mmlu": len(mmlu), "gsm8k": len(gsm8k), "reasoning": len(reasoning)},
        "results": [
            {
                "personality": r.personality_name,
                "traits": r.traits,
                "mmlu_score": r.mmlu_score,
                "gsm8k_score": r.gsm8k_score,
                "reasoning_score": r.reasoning_score,
                "average_score": r.average_score,
                "delta_vs_baseline": r.average_score - baseline_avg,
                "time": r.total_time
            }
            for r in results
        ]
    }

    results_file = output_dir / f"personality_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Update README with markdown
    md = f"""
## Personality A/B Benchmark Results

**Model**: `{model_name}`
**Date**: {datetime.now().strftime('%Y-%m-%d')}

### Performance by Personality Archetype

| Personality | MMLU | GSM8K | Reasoning | Average | vs Baseline |
|-------------|------|-------|-----------|---------|-------------|
"""
    for r in results:
        delta = r.average_score - baseline_avg
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        status = "[+]" if delta > 2 else "[-]" if delta < -2 else "[=]"
        md += f"| **{r.personality_name}** | {r.mmlu_score:.1f}% | {r.gsm8k_score:.1f}% | {r.reasoning_score:.1f}% | {r.average_score:.1f}% | {delta_str} {status} |\n"

    # Best performer
    best = max(results, key=lambda x: x.average_score)
    md += f"\n> **Best performing personality**: `{best.personality_name}` with {best.average_score:.1f}% average\n"

    md_file = output_dir / "personality_results.md"
    md_file.write_text(md, encoding='utf-8')

    print(f"\n[OK] Results saved to: {results_file}")
    print(f"[OK] Markdown saved to: {md_file}")

    # Update README
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding='utf-8')
        marker_start = "<!-- PERSONALITY_RESULTS_START -->"
        marker_end = "<!-- PERSONALITY_RESULTS_END -->"

        if marker_start in content:
            import re
            pattern = f"{marker_start}.*?{marker_end}"
            md += f"\nRun benchmarks: `python scripts/run_personality_benchmark.py --model {model_name} --samples {max_samples}`"
            replacement = f"{marker_start}\n{md}\n{marker_end}"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            readme_path.write_text(content, encoding='utf-8')
            print("[OK] README.md updated")


async def main():
    parser = argparse.ArgumentParser(description="Run personality A/B benchmarks")
    parser.add_argument("--model", default="tinyllama:latest", help="Ollama model")
    parser.add_argument("--samples", type=int, default=20, help="Samples per benchmark")

    args = parser.parse_args()
    await run_personality_benchmark(args.model, args.samples)


if __name__ == "__main__":
    asyncio.run(main())
