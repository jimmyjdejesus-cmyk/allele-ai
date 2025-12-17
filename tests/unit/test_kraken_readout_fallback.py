import builtins

import numpy as np

from phylogenic.kraken_lnn import KrakenLNN


def test_train_readout_uses_numpy_fallback_when_sklearn_missing(monkeypatch):
    rng = np.random.RandomState(0)
    kraken = KrakenLNN(reservoir_size=3, random_state=rng)

    n_samples = 30
    X = rng.randn(n_samples, kraken.reservoir_size)

    true_w = np.array([0.5, -1.2, 2.0])
    true_bias = 0.3
    y = X @ true_w + true_bias

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("No sklearn available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    res = kraken.train_readout(X, y, method="ridge", alpha=1e-6)
    assert "numpy" in res["method"] or "numpy" in getattr(kraken, "readout_method", "")

    # weights and bias should approximate the true values
    assert np.allclose(kraken.readout_weights, true_w, atol=1e-5)
    assert abs(kraken.readout_bias - true_bias) < 1e-5
