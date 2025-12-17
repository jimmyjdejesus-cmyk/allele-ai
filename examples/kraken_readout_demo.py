"""
Kraken LNN Readout Demo
=======================

This script demonstrates how to:
1. Initialize a Kraken LNN with a reservoir.
2. Process a sequence of data and store states in Temporal Memory.
3. Train a linear readout (supervised learning) from the stored memory
    to predict a target.
4. Save and load the trained readout model.
5. Make predictions on new data.

The task is a simple synthetic regression: y = sum(input_window) + noise.
"""

import asyncio
import logging
import os
import tempfile

import numpy as np

from phylogenic.kraken_lnn import KrakenLNN

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_sequence(length: int, seed: int = 42):
    """Generate a random sequence and corresponding targets."""
    rng = np.random.RandomState(seed)
    # Input: random sequence
    inputs = rng.randn(length)

    # Target: Moving average (window 3) + noise
    # This requires memory of past inputs to solve
    targets = []
    for i in range(length):
        window = inputs[max(0, i - 2) : i + 1]
        target = np.sum(window) + 0.01 * rng.randn()
        targets.append(target)

    return inputs.tolist(), targets


async def run_demo():
    logger.info("Initializing Kraken LNN...")
    # Create Kraken with a small reservoir
    kraken = KrakenLNN(
        reservoir_size=50,
        connectivity=0.2,
        memory_buffer_size=1000,
        instance_name="demo_kraken",
    )

    logger.info("Generating training data...")
    train_len = 200
    train_inputs, train_targets = generate_synthetic_sequence(train_len, seed=101)

    logger.info("Processing training sequence and populating memory...")
    # Process inputs one by one to populate memory
    # We attach the 'target' label to the memory entry for training
    for i, val in enumerate(train_inputs):
        # Process input (updates reservoir state)
        # We disable memory consolidation for this simple demo to keep raw states
        await kraken.process_sequence([val], memory_consolidation=False)

        # Manually add the target label to the last memory entry
        # In a real app, this might happen via feedback or delayed reward
        if kraken.temporal_memory._count > 0:
            # Get the latest memory (at head-1)
            latest_idx = (
                kraken.temporal_memory._head - 1
            ) % kraken.temporal_memory.buffer_size
            memory = kraken.temporal_memory.memories[latest_idx]
            if memory:
                memory["target_label"] = train_targets[i]

    logger.info(f"Memory populated with {len(kraken.temporal_memory)} entries.")

    logger.info("Training readout from memory...")

    # Define how to extract the label from a memory entry
    def label_extractor(mem):
        return mem.get("target_label", 0.0)

    # Train the readout
    # This uses the stored reservoir states (X) and extracted labels (y)
    metrics = kraken.train_readout_from_memory(
        label_extractor=label_extractor, method="ridge", alpha=0.5
    )

    logger.info(f"Training complete. Metrics: {metrics}")
    logger.info(f"Readout weights shape: {kraken.readout_weights.shape}")

    # --- Persistence Demo ---
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "kraken_readout.pkl")
        logger.info(f"Saving readout model to {save_path}...")
        kraken.save_readout(save_path)

        logger.info("Loading readout model into a new Kraken instance...")
        kraken_new = KrakenLNN(reservoir_size=50, connectivity=0.2)
        kraken_new.load_readout(save_path)

        # Verify weights match
        if np.allclose(kraken.readout_weights, kraken_new.readout_weights):
            logger.info("SUCCESS: Loaded weights match original weights.")
        else:
            logger.error("FAILURE: Loaded weights do not match!")

    # --- Prediction Demo ---
    logger.info("Running evaluation on new test data...")
    test_len = 50
    test_inputs, test_targets = generate_synthetic_sequence(test_len, seed=202)

    predictions = []

    # Use the trained kraken to predict
    # Note: We must process the sequence to update the reservoir state!
    mse_accum = 0.0

    for i, val in enumerate(test_inputs):
        # Update state
        await kraken.process_sequence([val], memory_consolidation=False)

        # Predict using the current state
        pred = kraken.readout_predict()
        predictions.append(pred)

        actual = test_targets[i]
        mse_accum += (pred - actual) ** 2

    mse = mse_accum / test_len
    logger.info(f"Test MSE: {mse:.4f}")

    # Show a few examples
    logger.info("Sample predictions:")
    for i in range(5):
        logger.info(
            "  Input: %0.2f -> Pred: %0.2f (Actual: %0.2f)",
            test_inputs[i],
            predictions[i],
            test_targets[i],
        )

    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_demo())
