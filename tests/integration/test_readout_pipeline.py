import numpy as np

from phylogenic.kraken_lnn import KrakenLNN


def test_readout_pipeline_train_save_load(tmp_path):
    rng = np.random.RandomState(0)
    reservoir_size = 5
    kraken = KrakenLNN(
        reservoir_size=reservoir_size, memory_buffer_size=50, random_state=rng
    )

    n_samples = 12
    true_w = np.array([0.7, -0.4, 1.1, -0.2, 0.9])
    true_bias = -0.15

    for _ in range(n_samples):
        state = rng.randn(reservoir_size)
        label = float(state @ true_w + true_bias)
        kraken.temporal_memory.add_memory({"reservoir_state": state, "label": label})

    res = kraken.train_readout_from_memory(
        lambda m: m["label"], min_samples=10, method="ridge", alpha=1e-6
    )
    assert res["n_samples"] == n_samples
    assert np.allclose(kraken.readout_weights, true_w, atol=1e-5)
    assert abs(kraken.readout_bias - true_bias) < 1e-5

    p = tmp_path / "readout.pkl"
    kraken.save_readout(str(p))
    assert p.exists()

    k2 = KrakenLNN(reservoir_size=reservoir_size)
    k2.load_readout(str(p))
    assert np.allclose(kraken.readout_weights, k2.readout_weights)
