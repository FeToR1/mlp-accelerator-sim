import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

import torch


def fp32(tensor, dimensions, name):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != dimensions:
        raise ValueError(f"{name}: expected a {dimensions}D tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name}: expected floating point values")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: values must be finite")
    return tensor


def write_values(stream, tensor):
    for row in tensor.reshape(-1, tensor.shape[-1]).tolist():
        stream.write(" ".join(format(value, ".9g") for value in row) + "\n")


def export_weights(state, path, names=None):
    if not isinstance(state, Mapping):
        raise ValueError("Save model.state_dict(), not the whole model")
    for key in ("state_dict", "model_state_dict"):
        if key in state:
            state = state[key]
            break
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must contain a state_dict")
    if names is None:
        names = [
            key[:-7] if key.endswith(".weight") else ""
            for key in state if key == "weight" or key.endswith(".weight")
        ]
    if not names or len(set(names)) != len(names):
        raise ValueError("Expected at least one Linear layer, without duplicates")
    layers, used, widths = [], set(), []
    for name in names:
        prefix = name + "." if name else ""
        weight_key, bias_key = prefix + "weight", prefix + "bias"
        if weight_key not in state:
            raise ValueError(f"Missing weight: {weight_key}")
        weight = fp32(state[weight_key], 2, weight_key)
        n, k = weight.shape
        if not k or not n or (widths and k != widths[-1]):
            raise ValueError(f"{weight_key}: layer dimensions do not match")
        if bias_key in state:
            bias = fp32(state[bias_key], 1, bias_key)
        else:
            bias = torch.zeros(n)
        if bias.shape != (n,):
            raise ValueError(f"{bias_key}: expected {n} values")
        used.add(weight_key)
        if bias_key in state:
            used.add(bias_key)
        if not widths:
            widths.append(k)
        widths.append(n)
        layers.append((weight.t().contiguous(), bias))
    if set(state) != used:
        raise ValueError(
            f"Only Linear weights are supported; unused keys: {sorted(set(state) - used)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"MLP_WEIGHTS_V1 {len(layers)}\n")
        for weight, bias in layers:
            stream.write(f"{weight.shape[0]} {weight.shape[1]}\n")
            write_values(stream, weight)
            write_values(stream, bias)
    return widths


def export_input(tensor, path, width):
    tensor = fp32(tensor, 2, "input")
    if tensor.shape[0] <= 0 or tensor.shape[1] != width:
        raise ValueError(f"Input must have shape [batch, {width}]")
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"MLP_INPUT_V1 {tensor.shape[0]} {width}\n")
        write_values(stream, tensor)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Linear weights for the simulator")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", nargs="+", help="Linear module names in forward order")
    parser.add_argument("--input", type=Path, help="A saved input tensor [batch, input width]")
    args = parser.parse_args()
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        widths = export_weights(state, args.output, args.layers)
        print(f"Weights: {args.output}; network: {'->'.join(map(str, widths))}")
        if args.input:
            tensor = torch.load(args.input, map_location="cpu", weights_only=True)
            input_path = args.output.with_suffix(".input.txt")
            export_input(tensor, input_path, widths[0])
            print(f"Input: {input_path}; batch: {tensor.shape[0]}")
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
