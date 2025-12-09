#!/usr/bin/env python3
"""
Basic usage example for Abe-NLP SDK.

This example demonstrates:
- Creating a conversational genome
- Setting up an agent
- Basic chat interaction
"""

import asyncio
from allele import ConversationalGenome, create_agent, AgentConfig


async def main():
    """Run basic usage example."""
    print("🧬 Abe-NLP Basic Usage Example\n")
    print("="*50)
    
    # Step 1: Create a genome with specific traits
    print("\n📊 Step 1: Creating conversational genome...")
    genome = ConversationalGenome(
        genome_id="example_agent_001",
        traits={
            'empathy': 0.85,              # High empathy
            'engagement': 0.75,           # Good engagement
            'technical_knowledge': 0.90,  # Expert technical knowledge
            'creativity': 0.65,           # Moderately creative
            'conciseness': 0.80,          # Concise responses
            'context_awareness': 0.85,    # Strong context retention
            'adaptability': 0.70,         # Adapts well
            'personability': 0.80         # Friendly personality
        }
    )
    
    print(f"   ✓ Genome ID: {genome.genome_id}")
    print(f"   ✓ Traits: {genome.traits}")
    
    # Step 2: Configure agent
    print("\n⚙️  Step 2: Configuring agent...")
    config = AgentConfig(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
        memory_enabled=True,
        evolution_enabled=True,
        kraken_enabled=True
    )
    
    print(f"   ✓ Model: {config.model_name}")
    print(f"   ✓ Temperature: {config.temperature}")
    print(f"   ✓ Kraken LNN: {'Enabled' if config.kraken_enabled else 'Disabled'}")
    
    # Step 3: Create agent
    print("\n🤖 Step 3: Creating agent...")
    agent = await create_agent(genome, config)
    
    print(f"   ✓ Agent initialized: {agent.is_initialized}")
    
    # Step 4: Chat with agent
    print("\n💬 Step 4: Starting conversation...")
    print("\n" + "="*50)
    
    messages = [
        "Hello! Can you introduce yourself?",
        "What are your strengths?",
        "Tell me about your technical capabilities."
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n[User #{i}]: {message}")
        print(f"[Agent]: ", end='')
        
        async for response_chunk in agent.chat(message):
            print(response_chunk, end='')
        
        print()  # New line after response
    
    print("\n" + "="*50)
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("  - Try examples/evolution.py for genome evolution")
    print("  - Try examples/kraken_lnn.py for neural network processing")
    print("  - See docs/ for full API documentation")


if __name__ == "__main__":
    asyncio.run(main())

