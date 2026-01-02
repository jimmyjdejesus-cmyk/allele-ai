import numpy as np

from phylogenic.kraken_lnn import KrakenLNN


def generate_synthetic_data(n_samples: int, reservoir_size: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    # Random reservoir states
    X = rng.randn(n_samples, reservoir_size)
    # True linear mapping
    true_w = rng.randn(reservoir_size)
    y = X @ true_w + 0.01 * rng.randn(n_samples)
    return X, y, true_w


def test_train_readout_ridge_numpy_fallback():
    n_samples = 100
    reservoir_size = 20
    X, y, true_w = generate_synthetic_data(n_samples, reservoir_size)

    kraken = KrakenLNN(reservoir_size=reservoir_size)
    res = kraken.train_readout(X, y, method="ridge", alpha=1.0)

    assert "mse" in res
    preds = X @ kraken.readout_weights + kraken.readout_bias
    mse = float(np.mean((preds - y) ** 2))
    assert mse <= res["mse"] + 1e-8


def test_train_readout_from_memory_and_persistence(tmp_path):
    n_samples = 50
    reservoir_size = 10
    X, y, _ = generate_synthetic_data(n_samples, reservoir_size, seed=123)

    kraken = KrakenLNN(reservoir_size=reservoir_size)

    # Populate memory with reservoir_state and synthetic labels
    for i in range(n_samples):
        kraken.temporal_memory.add_memory(
            {"reservoir_state": X[i].tolist(), "label": float(y[i])}
        )

    def label_extractor(m):
        return m["label"]

    res = kraken.train_readout_from_memory(
        label_extractor, min_samples=10, method="ridge", alpha=0.1
    )
    assert res["n_samples"] == n_samples

    # Save and load model
    p = tmp_path / "readout.pkl"
    kraken.save_readout(str(p))

    # Create new Kraken and load
    k2 = KrakenLNN(reservoir_size=reservoir_size)
    k2.load_readout(str(p))
    assert k2.readout_weights is not None
    preds_orig = X @ kraken.readout_weights + kraken.readout_bias
    preds_loaded = X @ k2.readout_weights + k2.readout_bias
    assert np.allclose(preds_orig, preds_loaded)
