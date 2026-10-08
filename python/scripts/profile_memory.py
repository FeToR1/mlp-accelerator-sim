from pathlib import Path
from tempfile import TemporaryDirectory

import sys

import memray
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "build" / "python"))

from nn import MLP, SGD

batch_size = 32
architecture = [784, 128, 10]
lr = 0.3
seed = 42


class MeanSquaredError:
    def __call__(self, targets, outputs):
        return np.mean((outputs - targets) ** 2)

    def grad(self, targets, outputs):
        return (2 / targets.size) * (outputs - targets)


data_dir = project_root / "data/MNIST/raw"
with (data_dir / "train-images-idx3-ubyte").open("rb") as file:
    header = np.frombuffer(file.read(16), dtype=">u4")
    pixels = int(header[2]) * int(header[3])
    images = np.frombuffer(file.read(batch_size * pixels), dtype=np.uint8)
    images = images.reshape(batch_size, pixels)

with (data_dir / "train-labels-idx1-ubyte").open("rb") as file:
    file.seek(8)
    labels = np.frombuffer(file.read(batch_size), dtype=np.uint8)


def measure(mode):
    with TemporaryDirectory(prefix="mlp-memray-") as directory:
        capture = Path(directory) / "capture.bin"
        with memray.Tracker(capture, native_traces=True, trace_python_allocators=True):
            x = images.astype(np.float32)
            x /= np.float32(255)
            model = MLP(architecture, seed=seed)
            if mode == "training":
                targets = np.eye(architecture[-1], dtype=np.float32)[labels]
                criterion = MeanSquaredError()
                optimizer = SGD(model.parameters(), lr=lr)

            for _ in range(5):
                if mode == "inference":
                    outputs = model.predict(x)
                else:
                    optimizer.zero_grad()
                    outputs = model.forward(x)
                    loss = criterion(targets, outputs)
                    model.backward(criterion.grad(targets, outputs))
                    optimizer.step()

        with memray.FileReader(capture) as reader:
            peak_bytes = sum(record.size for record in reader.get_high_watermark_allocation_records())

    print(f"{mode}: batch={batch_size}, peak={peak_bytes} bytes ({peak_bytes / 1024 ** 2:.3f} MiB)")


measure("inference")
measure("training")
