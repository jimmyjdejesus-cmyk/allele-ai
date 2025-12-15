#!/usr/bin/env python3
"""
Simple test script for Ollama LLM integration.

Run this to test the new Ollama support without needing API keys.
"""

import asyncio

from allele import AgentConfig, ConversationalGenome, NLPAgent


async def test_ollama_agent():
    """Test Ollama integration with a simple conversational agent."""

    # Create genome with personality traits
    genome = ConversationalGenome("ollama_test", {
        'empathy': 0.8,
        'engagement': 0.7,
        'technical_knowledge': 0.6,
        'creativity': 0.8,
        'conciseness': 0.7,
        'context_awareness': 0.8,
        'adaptability': 0.7,
        'personability': 0.9
    })

    # Configure for Ollama (no API key needed)
    config = AgentConfig(
        llm_provider="ollama",        # NEW: Ollama support!
        model_name="llama2:latest",   # Exact model name from Ollama
        temperature=0.7,
        max_tokens=1024,
        streaming=True,
        fallback_to_mock=True,        # Enable fallback if Ollama is down
        request_timeout=30
    )

    print("Testing Ollama Integration with Allele SDK")
    print(f"   Provider: {config.llm_provider}")
    print(f"   Model: {config.model_name}")
    print(f"   Genome: {genome.genome_id}")
    print(f"   Traits: {list(genome.traits.keys())[:4]}...")
    print()

    try:
        # Create agent
        print("Creating agent...")
        agent = NLPAgent(genome, config)

        # Initialize (connects to Ollama)
        print("Initializing agent (connecting to Ollama)...")
        await agent.initialize()

        print("Agent initialized successfully!")
        print()

        # Test chat
        test_messages = [
            "Hello! Can you tell me about yourself?",
            "What are your main personality traits?",
            "How does your genome influence your responses?"
        ]

        for i, message in enumerate(test_messages, 1):
            print(f"User {i}: {message}")
            print("Agent:", end=" ")

            response_chunks = []
            async for chunk in agent.chat(message):
                print(chunk, end="")
                response_chunks.append(chunk)

            print()  # New line after response
            print(f"   Response length: {len(''.join(response_chunks))} chars")
            print()

        # Show metrics
        metrics = await agent.get_metrics()
        print("Final Metrics:")
        print(f"   Total requests: {metrics['performance']['total_requests']}")
        print(f"   Successful: {metrics['performance']['successful_requests']}")
        print(f"   Average latency: {metrics['performance']['average_latency_ms']:.3f}ms")
        print(f"   Provider: {metrics['config']['provider']}")
        print(f"   Model: {metrics['config']['model']}")

        await agent.close()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: 'ollama serve'")
        print("Install models: 'ollama pull llama2'")
        return False

    print("SUCCESS: Ollama integration test completed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_ollama_agent())
