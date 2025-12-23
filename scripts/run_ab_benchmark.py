#!/usr/bin/env python3
"""
A/B Benchmark Runner for Phylogenic AI Genomes

Compares baseline LLM performance against phylogenic-enhanced models.

Group A (Baseline): Raw LLM responses without genome enhancement
Group B (Phylogenic): LLM with genetic personality traits and optimized prompting

Usage:
    python scripts/run_ab_benchmark.py --model gemma3:1b --samples 50
"""

# ARCHIMEDES-Ω: A/B Benchmark Runner
# SUBSTRATE: Hybrid (Ollama LLM + Genetic Genomes)
# PHYSICS_CONSTRAINT: Inference latency bounds

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phylogenic.llm_client import LLMConfig
from src.phylogenic.llm_ollama import OllamaClient
from src.phylogenic.genome import ConversationalGenome
from phylogenic.benchmark.utils import check_answer


@dataclass
class ABResult:
    """Container for A/B benchmark comparison results."""
    benchmark_name: str
    baseline_score: float
    phylogenic_score: float
    delta: float
    delta_percent: float
    baseline_time: float
    phylogenic_time: float
    samples_evaluated: int
    status: str  # "improved", "degraded", "neutral"


@dataclass
class ABBenchmarkSuite:
    """Complete A/B benchmark suite results."""
    model: str
    timestamp: str
    total_baseline_score: float = 0.0
    total_phylogenic_score: float = 0.0
    total_delta: float = 0.0
    results: List[ABResult] = field(default_factory=list)
    genome_config: Dict[str, float] = field(default_factory=dict)


class BaselineModel:
    """Baseline model wrapper - raw LLM without genome enhancement."""
    
    def __init__(self, client: OllamaClient):
        self.client = client
        self.name = "baseline"
    
    async def generate(self, prompt: str) -> str:
        """Generate response without any enhancement."""
        messages = [{"role": "user", "content": prompt}]
        response = ""
        async for chunk in self.client.chat_completion(messages, stream=False):
            response += chunk
        return response.strip()


class PhylogenicModel:
    """Phylogenic-enhanced model with genetic genome traits."""
    
    def __init__(self, client: OllamaClient, genome: ConversationalGenome):
        self.client = client
        self.genome = genome
        self.name = "phylogenic"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from genome traits."""
        traits = self.genome.traits
        
        # Map trait values to behavioral descriptors
        trait_descriptions = []
        
        if traits.get("empathy", 0.5) > 0.7:
            trait_descriptions.append("Show deep understanding and emotional intelligence")
        if traits.get("technical_knowledge", 0.5) > 0.7:
            trait_descriptions.append("Provide technically accurate and detailed explanations")
        if traits.get("creativity", 0.5) > 0.7:
            trait_descriptions.append("Think creatively and offer novel perspectives")
        if traits.get("conciseness", 0.5) > 0.7:
            trait_descriptions.append("Be direct and concise in responses")
        if traits.get("context_awareness", 0.5) > 0.7:
            trait_descriptions.append("Maintain strong awareness of context")
        if traits.get("adaptability", 0.5) > 0.7:
            trait_descriptions.append("Adapt communication style to the task")
        if traits.get("engagement", 0.5) > 0.7:
            trait_descriptions.append("Be engaging and maintain conversational flow")
        if traits.get("personability", 0.5) > 0.7:
            trait_descriptions.append("Be friendly and approachable")
        
        system_prompt = """You are an AI assistant with evolved personality traits optimized for high performance.

Your behavioral guidelines:
"""
        for desc in trait_descriptions:
            system_prompt += f"- {desc}\n"
        
        system_prompt += """
When answering questions:
1. Analyze the question carefully before responding
2. For multiple choice questions, state only the letter (A, B, C, or D)
3. For math problems, show reasoning then give the final answer
4. For code problems, write clean, working code
"""
        return system_prompt
    
    async def generate(self, prompt: str) -> str:
        """Generate response with genome-enhanced prompting."""
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = ""
        async for chunk in self.client.chat_completion(messages, stream=False):
            response += chunk
        return response.strip()


class SimpleBenchmark:
    """Simple benchmark implementation for A/B testing."""
    
    def __init__(self, name: str, samples: List[Dict[str, Any]]):
        self.name = name
        self.samples = samples
    
    async def evaluate(self, model: Any) -> tuple[float, int, float]:
        """Evaluate model and return (score, correct_count, time_taken)."""
        correct = 0
        total = len(self.samples)
        start_time = time.time()
        
        for sample in self.samples:
            try:
                prompt = sample["prompt"]
                expected = sample["expected"]
                
                response = await model.generate(prompt)
                
                # Check if response contains expected answer
                if self._check_answer(response, expected):
                    correct += 1
            except Exception as e:
                print(f"  Error evaluating sample: {e}")
                continue
        
        time_taken = time.time() - start_time
        score = (correct / total) * 100 if total > 0 else 0
        return score, correct, time_taken
    
    def _check_answer(self, response: str, expected: str) -> bool:
        """Delegate to central `check_answer` utility."""
        return check_answer(response, expected)


def create_mmlu_samples(max_samples: int = 50) -> List[Dict[str, Any]]:
    """Create MMLU-style benchmark samples."""
    samples = [
        {
            "prompt": "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the chemical symbol for gold?\nA. Go\nB. Gd\nC. Au\nD. Ag\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Who wrote 'Romeo and Juliet'?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the largest organ in the human body?\nA. Heart\nB. Brain\nC. Liver\nD. Skin\nAnswer:",
            "expected": "D"
        },
        {
            "prompt": "Question: What is the speed of light in vacuum?\nA. 300,000 km/s\nB. 150,000 km/s\nC. 450,000 km/s\nD. 600,000 km/s\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Question: Which gas do plants primarily absorb from the atmosphere?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: In what year did World War II end?\nA. 1943\nB. 1944\nC. 1945\nD. 1946\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the main function of red blood cells?\nA. Fight infection\nB. Carry oxygen\nC. Clot blood\nD. Digest food\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which element has atomic number 1?\nA. Helium\nB. Hydrogen\nC. Lithium\nD. Carbon\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the derivative of x²?\nA. x\nB. 2x\nC. x²\nD. 2\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which continent is the Sahara Desert located on?\nA. Asia\nB. Australia\nC. Africa\nD. South America\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Who developed the theory of relativity?\nA. Isaac Newton\nB. Albert Einstein\nC. Niels Bohr\nD. Max Planck\nAnswer:",
            "expected": "B"
        },
    ]
    return samples[:max_samples]


def create_gsm8k_samples(max_samples: int = 20) -> List[Dict[str, Any]]:
    """Create GSM8K-style math benchmark samples."""
    samples = [
        {
            "prompt": "Problem: If a store sells 5 apples for $2, how much would 15 apples cost?\nAnswer (just the number):",
            "expected": "6"
        },
        {
            "prompt": "Problem: A train travels 120 miles in 2 hours. What is its speed in miles per hour?\nAnswer (just the number):",
            "expected": "60"
        },
        {
            "prompt": "Problem: If John has 24 candies and gives away 1/3 of them, how many does he have left?\nAnswer (just the number):",
            "expected": "16"
        },
        {
            "prompt": "Problem: A rectangle has length 8 and width 5. What is its area?\nAnswer (just the number):",
            "expected": "40"
        },
        {
            "prompt": "Problem: If 3x + 7 = 22, what is x?\nAnswer (just the number):",
            "expected": "5"
        },
        {
            "prompt": "Problem: A book costs $15. With a 20% discount, what is the new price?\nAnswer (just the number):",
            "expected": "12"
        },
        {
            "prompt": "Problem: What is 15% of 200?\nAnswer (just the number):",
            "expected": "30"
        },
        {
            "prompt": "Problem: If a car uses 5 gallons of gas to travel 150 miles, how many miles per gallon does it get?\nAnswer (just the number):",
            "expected": "30"
        },
        {
            "prompt": "Problem: A classroom has 28 students. If 7 are absent, what fraction of the class is present? (answer as decimal)\nAnswer:",
            "expected": "0.75"
        },
        {
            "prompt": "Problem: What is the sum of all integers from 1 to 10?\nAnswer (just the number):",
            "expected": "55"
        },
    ]
    return samples[:max_samples]


def create_reasoning_samples(max_samples: int = 15) -> List[Dict[str, Any]]:
    """Create commonsense reasoning benchmark samples."""
    samples = [
        {
            "prompt": "Question: The man put the milk in the refrigerator because:\nA. It was empty\nB. It needed to stay cold\nC. He was hungry\nD. The door was open\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: After the rain stopped, the streets were:\nA. Dry\nB. Wet\nC. Hot\nD. Dark\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: The bird flew south for the winter because:\nA. It was bored\nB. It wanted to see new places\nC. The weather was getting cold\nD. It was lost\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: She brought an umbrella because:\nA. It was sunny\nB. She wanted exercise\nC. Rain was forecast\nD. She was cold\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: The baby started crying, so the mother:\nA. Went to sleep\nB. Left the house\nC. Comforted the baby\nD. Watched TV\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: The ice cream melted because:\nA. It was in the freezer\nB. It was a hot day\nC. Someone ate it\nD. It was chocolate\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: He studied all night because:\nA. He had nothing to do\nB. He had an exam the next day\nC. He was bored\nD. It was raining\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: The plant died because:\nA. It got too much sunlight\nB. It was never watered\nC. It grew too tall\nD. It was green\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: She wore sunglasses because:\nA. It was dark outside\nB. It was raining\nC. The sun was bright\nD. She was tired\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: The car wouldn't start because:\nA. It was new\nB. The battery was dead\nC. It was parked\nD. It was red\nAnswer:",
            "expected": "B"
        },
    ]
    return samples[:max_samples]


async def run_ab_benchmark(
    model_name: str = "gemma3:1b",
    max_samples: int = 50,
    output_dir: str = "benchmark_results"
) -> ABBenchmarkSuite:
    """Run complete A/B benchmark suite."""
    
    print("=" * 60)
    print("PHYLOGENIC AI GENOME A/B BENCHMARK")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Max samples per benchmark: {max_samples}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Initialize Ollama client
    config = LLMConfig(
        provider="ollama",
        model=model_name,
        temperature=0.1,  # Low temperature for deterministic answers
        max_tokens=256,
        timeout=120
    )
    
    client = OllamaClient(config)
    
    try:
        await client.initialize()
        print(f"[OK] Connected to Ollama ({model_name})")
    except Exception as e:
        print(f"[FAIL] Failed to connect to Ollama: {e}")
        print("\nMake sure Ollama is running and the model is available:")
        print(f"  ollama pull {model_name}")
        print("  ollama serve")
        sys.exit(1)
    
    # Create models
    baseline = BaselineModel(client)
    
    # Create optimized genome for benchmarking
    optimized_genome = ConversationalGenome(
        genome_id="benchmark_optimized_v1",
        traits={
            "empathy": 0.3,           # Lower for factual tasks
            "technical_knowledge": 0.95,  # High for accuracy
            "creativity": 0.4,        # Moderate
            "conciseness": 0.9,       # High for clear answers
            "context_awareness": 0.8,
            "adaptability": 0.7,
            "engagement": 0.3,        # Lower for focused responses
            "personability": 0.3      # Lower for factual tasks
        }
    )
    
    phylogenic = PhylogenicModel(client, optimized_genome)
    
    # Initialize results
    suite = ABBenchmarkSuite(
        model=model_name,
        timestamp=datetime.now().isoformat(),
        genome_config=optimized_genome.traits
    )
    
    # Define benchmarks
    benchmarks = [
        ("MMLU (Knowledge)", SimpleBenchmark("MMLU", create_mmlu_samples(max_samples))),
        ("GSM8K (Math)", SimpleBenchmark("GSM8K", create_gsm8k_samples(max_samples // 3))),
        ("HellaSwag (Reasoning)", SimpleBenchmark("HellaSwag", create_reasoning_samples(max_samples // 3))),
    ]
    
    # Run each benchmark
    for bench_name, benchmark in benchmarks:
        print(f"\n{'-' * 50}")
        print(f"Running: {bench_name}")
        print(f"Samples: {len(benchmark.samples)}")
        print("-" * 50)
        
        # Run baseline
        print("  [A] Baseline model...", end=" ", flush=True)
        baseline_score, baseline_correct, baseline_time = await benchmark.evaluate(baseline)
        print(f"{baseline_score:.1f}% ({baseline_correct}/{len(benchmark.samples)}) in {baseline_time:.1f}s")
        
        # Run phylogenic
        print("  [B] Phylogenic model...", end=" ", flush=True)
        phylogenic_score, phylogenic_correct, phylogenic_time = await benchmark.evaluate(phylogenic)
        print(f"{phylogenic_score:.1f}% ({phylogenic_correct}/{len(benchmark.samples)}) in {phylogenic_time:.1f}s")
        
        # Calculate delta
        delta = phylogenic_score - baseline_score
        delta_percent = (delta / baseline_score * 100) if baseline_score > 0 else 0
        
        if delta > 1:
            status = "improved"
            status_icon = "[+]"
        elif delta < -1:
            status = "degraded"
            status_icon = "[-]"
        else:
            status = "neutral"
            status_icon = "[=]"
        
        print(f"  {status_icon} Delta: {delta:+.1f}% ({status})")
        
        result = ABResult(
            benchmark_name=benchmark.name,
            baseline_score=baseline_score,
            phylogenic_score=phylogenic_score,
            delta=delta,
            delta_percent=delta_percent,
            baseline_time=baseline_time,
            phylogenic_time=phylogenic_time,
            samples_evaluated=len(benchmark.samples),
            status=status
        )
        suite.results.append(result)
    
    # Calculate totals
    if suite.results:
        suite.total_baseline_score = sum(r.baseline_score for r in suite.results) / len(suite.results)
        suite.total_phylogenic_score = sum(r.phylogenic_score for r in suite.results) / len(suite.results)
        suite.total_delta = suite.total_phylogenic_score - suite.total_baseline_score
    
    # Close client
    await client.close()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"ab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model": suite.model,
            "timestamp": suite.timestamp,
            "total_baseline_score": suite.total_baseline_score,
            "total_phylogenic_score": suite.total_phylogenic_score,
            "total_delta": suite.total_delta,
            "genome_config": suite.genome_config,
            "results": [
                {
                    "benchmark": r.benchmark_name,
                    "baseline_score": r.baseline_score,
                    "phylogenic_score": r.phylogenic_score,
                    "delta": r.delta,
                    "delta_percent": r.delta_percent,
                    "baseline_time": r.baseline_time,
                    "phylogenic_time": r.phylogenic_time,
                    "samples": r.samples_evaluated,
                    "status": r.status
                }
                for r in suite.results
            ]
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to: {results_file}")
    
    return suite


def print_results_matrix(suite: ABBenchmarkSuite) -> str:
    """Print and return results as markdown table."""
    
    print("\n" + "=" * 60)
    print("A/B BENCHMARK RESULTS MATRIX")
    print("=" * 60)
    
    # Build markdown table (using ASCII for console, emoji for markdown file)
    md = f"""
## Phylogenic Genome A/B Benchmark Results

**Model**: `{suite.model}`  
**Date**: {suite.timestamp[:10]}

### Performance Comparison

| Benchmark | Baseline | + Genome | Delta | Status |
|-----------|----------|----------|-------|--------|
"""
    
    for r in suite.results:
        if r.status == "improved":
            status_icon = "[+] Improved"
        elif r.status == "degraded":
            status_icon = "[-] Degraded"
        else:
            status_icon = "[=] Neutral"
        
        md += f"| **{r.benchmark_name}** | {r.baseline_score:.1f}% | {r.phylogenic_score:.1f}% | {r.delta:+.1f}% | {status_icon} |\n"
    
    # Add totals
    total_status = "[+]" if suite.total_delta > 0 else "[-]" if suite.total_delta < 0 else "[=]"
    md += f"| **AVERAGE** | **{suite.total_baseline_score:.1f}%** | **{suite.total_phylogenic_score:.1f}%** | **{suite.total_delta:+.1f}%** | {total_status} |\n"
    
    md += f"""
### Genome Configuration

| Trait | Value |
|-------|-------|
"""
    for trait, value in suite.genome_config.items():
        md += f"| {trait} | {value:.2f} |\n"
    
    md += f"""
### Interpretation

- **Baseline**: Raw LLM responses without enhancement
- **+ Genome**: LLM with phylogenic personality traits applied
- **Delta**: Performance difference (positive = improvement)

"""
    
    if suite.total_delta > 2:
        md += "> **Conclusion**: Phylogenic genome enhancement shows significant improvement over baseline.\n"
    elif suite.total_delta > 0:
        md += "> **Conclusion**: Phylogenic genome enhancement shows slight improvement over baseline.\n"
    elif suite.total_delta < -2:
        md += "> **Conclusion**: Baseline performs better - genome configuration may need tuning.\n"
    else:
        md += "> **Conclusion**: Performance is comparable between baseline and genome-enhanced models.\n"
    
    print(md)
    return md


async def main():
    parser = argparse.ArgumentParser(description="Run A/B benchmarks for Phylogenic AI Genomes")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model to test")
    parser.add_argument("--samples", type=int, default=50, help="Max samples per benchmark")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    parser.add_argument("--update-readme", action="store_true", help="Update README with results")
    
    args = parser.parse_args()
    
    # Run benchmarks
    suite = await run_ab_benchmark(
        model_name=args.model,
        max_samples=args.samples,
        output_dir=args.output
    )
    
    # Print results matrix
    markdown = print_results_matrix(suite)
    
    # Optionally update README
    if args.update_readme:
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text(encoding='utf-8')
            
            # Find or create benchmark section
            marker_start = "<!-- BENCHMARK_RESULTS_START -->"
            marker_end = "<!-- BENCHMARK_RESULTS_END -->"
            
            if marker_start in content:
                # Replace existing section
                import re
                pattern = f"{marker_start}.*?{marker_end}"
                replacement = f"{marker_start}\n{markdown}\n{marker_end}"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            else:
                # Add new section before ## Contributing
                insert_point = content.find("## Contributing")
                if insert_point != -1:
                    new_section = f"\n{marker_start}\n{markdown}\n{marker_end}\n\n"
                    content = content[:insert_point] + new_section + content[insert_point:]
            
            readme_path.write_text(content, encoding='utf-8')
            print(f"\n[OK] README.md updated with benchmark results")
    
    # Save markdown separately
    md_path = Path(args.output) / "results_matrix.md"
    md_path.write_text(markdown, encoding='utf-8')
    print(f"[OK] Markdown saved to: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
