"""Quick check of the original Eigen implementation after building the project."""
from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "build" / "python"))

from nn import MLP, SGD


def main():
    x = np.array([[0.0, 0.5, 1.0], [1.0, -0.5, 0.0]], dtype=np.float32)
    model = MLP([3, 4, 2], seed=42)
    prediction = model.predict(x)
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    np.testing.assert_array_equal(prediction, MLP([3, 4, 2], seed=42).predict(x))

    optimizer = SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    forward = model.forward(x)
    np.testing.assert_array_equal(prediction, forward)
    model.backward(np.ones_like(forward))
    optimizer.step()
    updated = model.predict(x)
    assert np.isfinite(updated).all()
    assert not np.array_equal(prediction, updated)
    print("Eigen reference: predict, forward, backward and SGD OK")


if __name__ == "__main__":
    main()
