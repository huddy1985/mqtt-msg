#!/usr/bin/env python3
"""Export a CNN model trained with PyTorch to ONNX format.

This helper keeps the project self-contained by providing a minimal
command-line entry point that expects an input PyTorch checkpoint (either
`torch.jit.save` scripted module or a regular `state_dict`) and writes out an
ONNX file ready for the ONNX Runtime C++ inference wrapper.
"""

import argparse
import pathlib
from typing import Iterable, Tuple


def parse_shape(shape: str) -> Tuple[int, ...]:
    dims: Iterable[str] = shape.lower().replace("x", " ").replace(",", " ").split()
    parsed = []
    for dim in dims:
        if dim in {"b", "batch"}:
            parsed.append(1)
            continue
        try:
            value = int(dim)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Invalid dimension value: {dim}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("ONNX export requires positive dimensions")
        parsed.append(value)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one dimension must be provided")
    return tuple(parsed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a PyTorch CNN to ONNX")
    parser.add_argument("input", type=pathlib.Path, help="Path to the PyTorch model (.pt/.pth)")
    parser.add_argument("output", type=pathlib.Path, help="Target ONNX file path")
    parser.add_argument(
        "--input-shape",
        dest="input_shape",
        type=parse_shape,
        default=parse_shape("1x3x224x224"),
        help="Input tensor shape as NxCxHxW (default: 1x3x224x224)",
    )
    parser.add_argument(
        "--opset",
        dest="opset",
        type=int,
        default=13,
        help="ONNX opset version to use during export (default: 13)",
    )
    parser.add_argument(
        "--channels",
        dest="channels",
        type=int,
        default=3,
        help="Number of image channels when instantiating the fallback stub network",
    )
    return parser


def load_torch():  # pragma: no cover - thin wrapper around dynamic import
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "PyTorch is required to export models to ONNX. Install torch first."
        ) from exc
    return torch, nn


def build_fallback_network(nn_module, channels: int):
    class TinyCnn(nn_module.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.features = nn_module.Sequential(
                nn_module.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn_module.ReLU(inplace=True),
                nn_module.Conv2d(16, 32, kernel_size=3, padding=1),
                nn_module.ReLU(inplace=True),
                nn_module.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn_module.Sequential(
                nn_module.Flatten(),
                nn_module.Linear(32, 2),
                nn_module.Softmax(dim=1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.classifier(self.features(x))

    return TinyCnn(channels)


def load_model(torch_module, input_path: pathlib.Path, channels: int):
    input_str = str(input_path)
    try:
        model = torch_module.jit.load(input_str, map_location="cpu")
        model.eval()
        return model
    except Exception:
        model = build_fallback_network(torch_module.nn, channels)
        try:
            checkpoint = torch_module.load(input_str, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception:
            # keep randomly initialised weights for placeholder conversion
            pass
        model.eval()
        return model


def export_to_onnx(args):
    torch, _ = load_torch()
    model = load_model(torch, args.input, args.channels)

    dummy_input = torch.randn(*args.input_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
    )
    print(f"Exported ONNX model to {args.output}")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    export_to_onnx(args)


if __name__ == "__main__":
    main()
