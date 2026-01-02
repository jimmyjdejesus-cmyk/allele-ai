#!/usr/bin/env python3
"""
Direct Ollama Benchmark Runner

Uses direct Ollama API calls instead of lm_eval to bypass compatibility issues.
Integrates with the matrix evaluation system.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phylogenic.benchmark.utils import check_answer, build_system_prompt, build_cot_prompt
from src.phylogenic.genome import ConversationalGenome
from src.phylogenic.llm_client import LLMConfig
from src.phylogenic.llm_ollama import OllamaClient

# Sample data functions (copied to avoid import dependencies)
def create_mmlu_samples(max_samples: int = 50):
    """Create MMLU-style benchmark samples."""
    samples = [
        {"prompt": "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:", "expected": "C"},
        {"prompt": "Question: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer:", "expected": "B"},
        {"prompt": "Question: What is the chemical symbol for gold?\nA. Go\nB. Gd\nC. Au\nD. Ag\nAnswer:", "expected": "C"},
        {"prompt": "Question: Who wrote 'Romeo and Juliet'?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain\nAnswer:", "expected": "B"},
        {"prompt": "Question: What is the largest organ in the human body?\nA. Heart\nB. Brain\nC. Liver\nD. Skin\nAnswer:", "expected": "D"},
        {"prompt": "Question: What is the speed of light in vacuum?\nA. 300,000 km/s\nB. 150,000 km/s\nC. 450,000 km/s\nD. 600,000 km/s\nAnswer:", "expected": "A"},
        {"prompt": "Question: Which gas do plants primarily absorb from the atmosphere?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nAnswer:", "expected": "C"},
        {"prompt": "Question: What is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:", "expected": "C"},
        {"prompt": "Question: In what year did World War II end?\nA. 1943\nB. 1944\nC. 1945\nD. 1946\nAnswer:", "expected": "C"},
        {"prompt": "Question: What is the main function of red blood cells?\nA. Fight infection\nB. Carry oxygen\nC. Clot blood\nD. Digest food\nAnswer:", "expected": "B"},
        {"prompt": "Question: Which element has atomic number 1?\nA. Helium\nB. Hydrogen\nC. Lithium\nD. Carbon\nAnswer:", "expected": "B"},
        {"prompt": "Question: What is the derivative of x²?\nA. x\nB. 2x\nC. x²\nD. 2\nAnswer:", "expected": "B"},
        {"prompt": "Question: Which continent is the Sahara Desert located on?\nA. Asia\nB. Australia\nC. Africa\nD. South America\nAnswer:", "expected": "C"},
        {"prompt": "Question: What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus\nAnswer:", "expected": "C"},
        {"prompt": "Question: Who developed the theory of relativity?\nA. Isaac Newton\nB. Albert Einstein\nC. Niels Bohr\nD. Max Planck\nAnswer:", "expected": "B"},
    ]
    return samples[:max_samples]

def create_gsm8k_samples(max_samples: int = 20):
    """Create GSM8K-style math benchmark samples."""
    samples = [
        {"prompt": "Problem: If a store sells 5 apples for $2, how much would 15 apples cost?\nAnswer (just the number):", "expected": "6"},
        {"prompt": "Problem: A train travels 120 miles in 2 hours. What is its speed in miles per hour?\nAnswer (just the number):", "expected": "60"},
        {"prompt": "Problem: If John has 24 candies and gives away 1/3 of them, how many does he have left?\nAnswer (just the number):", "expected": "16"},
        {"prompt": "Problem: A rectangle has length 8 and width 5. What is its area?\nAnswer (just the number):", "expected": "40"},
        {"prompt": "Problem: If 3x + 7 = 22, what is x?\nAnswer (just the number):", "expected": "5"},
        {"prompt": "Problem: A book costs $15. With a 20% discount, what is the new price?\nAnswer (just the number):", "expected": "12"},
        {"prompt": "Problem: What is 15% of 200?\nAnswer (just the number):", "expected": "30"},
        {"prompt": "Problem: If a car uses 5 gallons of gas to travel 150 miles, how many miles per gallon does it get?\nAnswer (just the number):", "expected": "30"},
        {"prompt": "Problem: What is the sum of all integers from 1 to 10?\nAnswer (just the number):", "expected": "55"},
    ]
    return samples[:max_samples]

def create_reasoning_samples(max_samples: int = 15):
    """Create commonsense reasoning benchmark samples."""
    samples = [
        {"prompt": "Question: The man put the milk in the refrigerator because:\nA. It was empty\nB. It needed to stay cold\nC. He was hungry\nD. The door was open\nAnswer:", "expected": "B"},
        {"prompt": "Question: After the rain stopped, the streets were:\nA. Dry\nB. Wet\nC. Hot\nD. Dark\nAnswer:", "expected": "B"},
        {"prompt": "Question: The bird flew south for the winter because:\nA. It was bored\nB. It wanted to see new places\nC. The weather was getting cold\nD. It was lost\nAnswer:", "expected": "C"},
    ]
    return samples[:max_samples]

def create_hellaswag_samples(max_samples: int = 20):
    """Create HellaSwag-style commonsense reasoning samples."""
    samples = [
        {
            "prompt": "Context: A person is at a coffee shop ordering a drink.\nA. They order a glass of water because they're thirsty.\nB. They order a coffee because they need caffeine.\nC. They order a salad because they're hungry.\nD. They order nothing and just sit down.\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Context: A person is walking their dog in the park.\nA. The dog starts chasing squirrels.\nB. The dog starts reading a newspaper.\nC. The dog starts flying in the sky.\nD. The dog starts cooking dinner.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is giving a presentation to a group.\nA. They use slides and speak clearly to the audience.\nB. They run away from the room screaming.\nC. They start singing opera loudly.\nD. They hide under the table.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is cooking dinner in the kitchen.\nA. They chop vegetables and stir the sauce.\nB. They throw the ingredients at the wall.\nC. They eat the raw ingredients directly.\nD. They use the food to paint a picture.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is trying to start their car in the morning.\nA. They put the key in the ignition and turn it.\nB. They sing a song to the car.\nC. They try to push the car to start it.\nD. They ask the car politely to start.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is reading a book in a library.\nA. They turn the pages quietly and focus on the text.\nB. They start shouting at the book.\nC. They throw the book across the room.\nD. They use the book as a pillow.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is washing dishes in the sink.\nA. They use soap and water to clean each dish.\nB. They throw the dishes out the window.\nC. They eat the dishes.\nD. They paint the dishes different colors.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is waiting for a bus at a bus stop.\nA. They check the schedule and look down the street.\nB. They start dancing on the bench.\nC. They try to fly away.\nD. They dig a hole in the ground.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is studying for an exam.\nA. They read their notes and practice problems.\nB. They set their notes on fire.\nC. They eat their textbook.\nD. They use their notes as confetti.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is watering plants in their garden.\nA. They use a watering can to give water to each plant.\nB. They pour gasoline on the plants.\nC. They try to eat the plants.\nD. They paint the plants different colors.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is fixing a bicycle tire.\nA. They remove the tire, patch the hole, and reinstall it.\nB. They throw the bicycle in a river.\nC. They try to ride the bicycle underwater.\nD. They use the bicycle as a musical instrument.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is setting up a tent for camping.\nA. They unfold the tent, stake it down, and secure the poles.\nB. They set the tent on fire.\nC. They try to wear the tent as clothing.\nD. They use the tent as a sailboat.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is making a sandwich for lunch.\nA. They get bread, add fillings, and cut it in half.\nB. They throw all the ingredients at the wall.\nC. They try to eat the plate instead.\nD. They use the sandwich as a hat.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is checking their email on a computer.\nA. They open their email client and read new messages.\nB. They throw the computer out the window.\nC. They try to eat the keyboard.\nD. They use the computer as a doorstop.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is folding laundry after it's been washed.\nA. They match socks, fold shirts, and organize by type.\nB. They set the laundry on fire.\nC. They try to wear all the clothes at once.\nD. They use the clothes as building materials.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is filling up their car with gas.\nA. They insert the nozzle, select the fuel type, and pump gas.\nB. They try to drink the gasoline.\nC. They set the gas station on fire.\nD. They use the gas pump as a microphone.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is organizing books on a bookshelf.\nA. They sort by author or topic and arrange them neatly.\nB. They throw all the books in the trash.\nC. They try to eat the books.\nD. They use the books as stepping stones.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is brushing their teeth before bed.\nA. They apply toothpaste, brush for two minutes, and rinse.\nB. They try to brush their hair with the toothbrush.\nC. They eat the toothpaste.\nD. They use the toothbrush as a paintbrush.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: Someone is packing a suitcase for a trip.\nA. They fold clothes, add toiletries, and organize items.\nB. They throw everything randomly into the suitcase.\nC. They try to wear the suitcase.\nD. They use the suitcase as a boat.\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Context: A person is feeding their pet cat.\nA. They open a can of cat food and put it in the bowl.\nB. They try to feed the cat chocolate.\nC. They set the cat food on fire.\nD. They use the cat food as paint.\nAnswer:",
            "expected": "A"
        },
    ]
    return samples[:max_samples]

def create_arc_easy_samples(max_samples: int = 20):
    """Create ARC-Easy style science reasoning samples."""
    samples = [
        {
            "prompt": "Question: What happens to water when it is heated to 100°C at sea level?\nA. It freezes\nB. It boils\nC. It condenses\nD. It evaporates slowly\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which of these is a mammal?\nA. Shark\nB. Dolphin\nC. Octopus\nD. Jellyfish\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What gas do plants absorb from the atmosphere during photosynthesis?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the primary function of the heart?\nA. To digest food\nB. To pump blood\nC. To filter waste\nD. To produce hormones\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which planet is closest to the Sun?\nA. Venus\nB. Earth\nC. Mercury\nD. Mars\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the chemical symbol for gold?\nA. Go\nB. Gd\nC. Au\nD. Ag\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the largest planet in our solar system?\nA. Earth\nB. Saturn\nC. Jupiter\nD. Neptune\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What process do plants use to make their own food?\nA. Respiration\nB. Photosynthesis\nC. Digestion\nD. Circulation\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the smallest unit of matter?\nA. Molecule\nB. Atom\nC. Cell\nD. Electron\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which gas makes up most of Earth's atmosphere?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Argon\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the speed of light in vacuum?\nA. 300,000 km/s\nB. 150,000 km/s\nC. 450,000 km/s\nD. 600,000 km/s\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Question: What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Which of these is a renewable energy source?\nA. Coal\nB. Natural gas\nC. Solar power\nD. Oil\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the main function of red blood cells?\nA. Fight infection\nB. Carry oxygen\nC. Clot blood\nD. Digest food\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the freezing point of water at sea level?\nA. 0°C\nB. 32°C\nC. 100°C\nD. -10°C\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Question: Which layer of the Earth is the thickest?\nA. Crust\nB. Mantle\nC. Outer core\nD. Inner core\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the primary source of energy for Earth?\nA. Wind\nB. The Sun\nC. Geothermal heat\nD. Tidal forces\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the most abundant element in the universe?\nA. Helium\nB. Carbon\nC. Hydrogen\nD. Oxygen\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Which of these is a type of rock formed from cooled magma?\nA. Sedimentary\nB. Metamorphic\nC. Igneous\nD. Limestone\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the process by which plants release water vapor?\nA. Photosynthesis\nB. Transpiration\nC. Respiration\nD. Evaporation\nAnswer:",
            "expected": "B"
        },
    ]
    return samples[:max_samples]

def create_truthfulqa_samples(max_samples: int = 20):
    """Create TruthfulQA-style truthfulness evaluation samples."""
    samples = [
        {
            "prompt": "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: How many days are in a week?\nA. 5\nB. 6\nC. 7\nD. 8\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Who wrote 'Romeo and Juliet'?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: In what year did World War II end?\nA. 1943\nB. 1944\nC. 1945\nD. 1946\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the largest ocean on Earth?\nA. Atlantic\nB. Indian\nC. Pacific\nD. Arctic\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: How many continents are there?\nA. 5\nB. 6\nC. 7\nD. 8\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the chemical symbol for water?\nA. H2O\nB. CO2\nC. NaCl\nD. O2\nAnswer:",
            "expected": "A"
        },
        {
            "prompt": "Question: Who painted the Mona Lisa?\nA. Vincent van Gogh\nB. Pablo Picasso\nC. Leonardo da Vinci\nD. Michelangelo\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the square root of 16?\nA. 2\nB. 4\nC. 8\nD. 16\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which country is home to the Eiffel Tower?\nA. Italy\nB. Spain\nC. France\nD. Germany\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the largest mammal in the world?\nA. Elephant\nB. Blue whale\nC. Giraffe\nD. Hippopotamus\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: How many sides does a triangle have?\nA. 2\nB. 3\nC. 4\nD. 5\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the capital of Japan?\nA. Seoul\nB. Beijing\nC. Tokyo\nD. Bangkok\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: Who developed the theory of relativity?\nA. Isaac Newton\nB. Albert Einstein\nC. Niels Bohr\nD. Max Planck\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: What is the freezing point of water in Fahrenheit?\nA. 0°F\nB. 32°F\nC. 100°F\nD. 212°F\nAnswer:",
            "expected": "B"
        },
        {
            "prompt": "Question: Which ocean is between America and Europe?\nA. Pacific\nB. Indian\nC. Atlantic\nD. Arctic\nAnswer:",
            "expected": "C"
        },
        {
            "prompt": "Question: What is the largest organ in the human body?\nA. Heart\nB. Brain\nC. Liver\nD. Skin\nAnswer:",
            "expected": "D"
        },
    ]
    return samples[:max_samples]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Benchmark sample data
BENCHMARK_SAMPLES = {
    "mmlu": create_mmlu_samples,
    "gsm8k": create_gsm8k_samples,
    "hellaswag": create_hellaswag_samples,
    "arc_easy": create_arc_easy_samples,
    "truthfulqa_mc2": create_truthfulqa_samples,
}


class GenomeModel:
    """Model with optional genome enhancement and COT support."""

    def __init__(self, client: OllamaClient, genome: ConversationalGenome = None, cot_mode: bool = False):
        self.client = client
        self.genome = genome
        self.cot_mode = cot_mode

    def _build_system_prompt(self) -> str:
        """Delegate prompt building to shared utility to avoid duplication."""
        if self.genome is None:
            return ""
        return build_system_prompt(self.genome.traits)

    async def generate(self, prompt: str) -> str:
        system_prompt = self._build_system_prompt()
        
        # Apply COT wrapper to user prompt if COT mode is enabled
        user_prompt = prompt
        if self.cot_mode:
            user_prompt = build_cot_prompt(prompt)

        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        response = ""
        async for chunk in self.client.chat_completion(messages, stream=False):
            response += chunk
        return response.strip()


class DirectOllamaBenchmarkRunner:
    """Direct Ollama API benchmark runner (bypasses lm_eval)."""

    def __init__(self, output_dir: str = "benchmark_results/direct_ollama"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_benchmark_task(
        self,
        model: str,
        tasks: List[str],
        limit: Optional[int] = None,
        num_fewshot: int = 0,
        traits: Optional[Dict[str, float]] = None,
        cot_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Run benchmark using direct Ollama API calls.
        
        Args:
            model: Model name (e.g., "llama3.2:3b")
            tasks: List of benchmark tasks (e.g., ["mmlu", "gsm8k"])
            limit: Maximum number of samples per task
            num_fewshot: Number of few-shot examples (not used in direct mode)
            traits: Optional personality traits dict
            cot_mode: If True, apply Chain of Thought prompting
        
        Returns:
            Results dict in lm_eval-compatible format
        """
        # Initialize Ollama client
        config = LLMConfig(
            provider="ollama",
            model=model,
            temperature=0.1,
            max_tokens=256,
            timeout=120
        )

        client = OllamaClient(config)
        
        try:
            await client.initialize()
            logger.info(f"Connected to Ollama ({model})")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return None

        # Create genome if traits provided
        genome = None
        if traits:
            genome = ConversationalGenome(
                genome_id=f"benchmark_{model}",
                traits=traits
            )

        # Create model wrapper
        model_wrapper = GenomeModel(client, genome=genome, cot_mode=cot_mode)

        # Run benchmarks
        results = {"results": {}}
        
        for task in tasks:
            logger.info(f"Running {task} benchmark...")
            
            # Get samples for this task
            if task not in BENCHMARK_SAMPLES:
                logger.warning(f"Unknown benchmark task: {task}, skipping")
                continue
            
            sample_func = BENCHMARK_SAMPLES[task]
            samples = sample_func(max_samples=limit or 50)
            
            if not samples:
                logger.warning(f"No samples available for {task}")
                continue

            # Evaluate samples
            correct = 0
            total = len(samples)
            
            for i, sample in enumerate(samples):
                try:
                    response = await model_wrapper.generate(sample["prompt"])
                    if check_answer(response, sample["expected"]):
                        correct += 1
                    
                    # Log progress every 10 samples
                    if (i + 1) % 10 == 0:
                        logger.debug(f"  {task}: {i + 1}/{total} samples processed")
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {i+1} for {task}: {e}")
                    continue

            # Calculate score
            score = (correct / total) * 100 if total > 0 else 0.0
            
            logger.info(f"  {task}: {correct}/{total} correct ({score:.1f}%)")
            
            # Store results in lm_eval-compatible format
            results["results"][task] = {
                "acc,none": score / 100.0,  # Convert percentage to decimal
                "acc": score / 100.0,
                "correct": correct,
                "total": total
            }

        return results

    def run_benchmark_task_sync(
        self,
        model: str,
        tasks: List[str],
        limit: Optional[int] = None,
        num_fewshot: int = 0,
        traits: Optional[Dict[str, float]] = None,
        cot_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for run_benchmark_task."""
        return asyncio.run(self.run_benchmark_task(
            model, tasks, limit, num_fewshot, traits, cot_mode
        ))


if __name__ == "__main__":
    # Quick test
    async def test():
        runner = DirectOllamaBenchmarkRunner()
        result = await runner.run_benchmark_task(
            model="llama3.2:3b",
            tasks=["mmlu"],
            limit=5
        )
        print(json.dumps(result, indent=2))

    asyncio.run(test())

