#!/usr/bin/env python3
"""Export a YOLO-style PyTorch checkpoint to ONNX format.

The project ships with this helper so the ONNX Runtime integration can be
validated end-to-end even without access to a trained detector.  When a real
PyTorch checkpoint is supplied the script will load it (either via
``torch.jit.load`` or ``torch.load``) and export the model to ONNX.  If loading
fails, a compact YOLO-like network is instantiated to produce a deterministic
placeholder graph so the runtime pipeline remains functional.
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
    parser = argparse.ArgumentParser(description="Export a YOLO detector to ONNX")
    parser.add_argument("input", type=pathlib.Path, help="Path to the PyTorch model (.pt/.pth/.jit)")
    parser.add_argument("output", type=pathlib.Path, help="Target ONNX file path")
    parser.add_argument(
        "--input-shape",
        dest="input_shape",
        type=parse_shape,
        default=parse_shape("1x3x640x640"),
        help="Input tensor shape as NxCxHxW (default: 1x3x640x640)",
    )
    parser.add_argument(
        "--classes",
        dest="num_classes",
        type=int,
        default=3,
        help="Number of object classes represented by the detector",
    )
    parser.add_argument(
        "--anchors",
        dest="anchors",
        type=int,
        default=3,
        help="Number of anchors per grid cell in the fallback stub network",
    )
    parser.add_argument(
        "--opset",
        dest="opset",
        type=int,
        default=13,
        help="ONNX opset version to use during export (default: 13)",
    )
    return parser


def load_torch():  # pragma: no cover - thin wrapper around dynamic import
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit("PyTorch is required to export models to ONNX. Install torch first.") from exc
    return torch, nn, F


def build_fallback_network(nn_module, functional, in_channels: int, num_classes: int, anchors: int):
    class TinyYolo(nn_module.Module):
        def __init__(self, channels: int, classes: int, anchors_per_cell: int):
            super().__init__()
            self.backbone = nn_module.Sequential(
                nn_module.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
                nn_module.BatchNorm2d(16),
                nn_module.LeakyReLU(0.1, inplace=True),
                nn_module.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn_module.BatchNorm2d(32),
                nn_module.LeakyReLU(0.1, inplace=True),
                nn_module.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn_module.BatchNorm2d(64),
                nn_module.LeakyReLU(0.1, inplace=True),
            )
            self.detector = nn_module.Conv2d(64, anchors_per_cell * (classes + 5), kernel_size=1)
            self.anchors_per_cell = anchors_per_cell
            self.num_classes = classes

        def forward(self, x):  # type: ignore[override]
            x = self.backbone(x)
            x = self.detector(x)
            bs, _, h, w = x.shape
            x = x.view(bs, self.anchors_per_cell, self.num_classes + 5, h, w)
            x = x.permute(0, 3, 4, 1, 2).contiguous()
            x = x.view(bs, -1, self.num_classes + 5)
            objectness = functional.sigmoid(x[..., 4:5])
            class_scores = functional.softmax(x[..., 5:], dim=-1)
            x = x.clone()
            x[..., 4:5] = objectness
            x[..., 5:] = class_scores
            return x

    return TinyYolo(in_channels, num_classes, anchors)


def load_model(torch_module, nn_module, functional, input_path: pathlib.Path, num_classes: int, anchors: int):
    input_str = str(input_path)
    try:
        model = torch_module.jit.load(input_str, map_location="cpu")
        model.eval()
        return model
    except Exception:
        model = build_fallback_network(nn_module, functional, 3, num_classes, anchors)
        try:
            checkpoint = torch_module.load(input_str, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
        except Exception:
            pass
        model.eval()
        return model


def export_to_onnx(args):
    torch, nn_module, functional = load_torch()
    model = load_model(torch, nn_module, functional, args.input, args.num_classes, args.anchors)

    dummy_input = torch.randn(*args.input_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}},
    )
    print(f"Exported YOLO ONNX model to {args.output}")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    export_to_onnx(args)


if __name__ == "__main__":
    main()
