"""Compare the C++ inference API with the unchanged original Python module."""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, sys.argv[2])
from nn import MLP

fixtures = json.loads(subprocess.check_output(
    [sys.argv[1], "--legacy-fixtures"], text=True))
max_error = 0.0
for fixture in fixtures:
    x = np.array(fixture["input"], dtype=np.float32)
    expected = MLP(fixture["sizes"], seed=42).predict(x)
    actual = np.array(fixture["output"], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    max_error = max(max_error, float(np.max(np.abs(actual - expected))))
print(f"Legacy parity: {len(fixtures)} topologies, max absolute error={max_error}")
