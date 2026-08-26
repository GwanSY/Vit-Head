#!/usr/bin/env python3
"""Protocol-aware ViT head reordering with a plaintext oracle.

Hugging Face stores linear weights as [output, input].  Reordering heads thus
permutes output rows (and bias entries) of Q/K/V and the matching input
columns of WO.  The operation is an exact parameter permutation, not a model
update.  Reports produced from a diagnostic calibration assignment are
explicitly non-final and cannot be exported as a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PATH_ORDER = {"softmax_mpc": 0, "relu_norm_mpc": 1, "he_linear": 2}
STAGES = ("q", "k", "v", "qk", "normalization", "av", "wo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assignment(path: Path, allow_diagnostic: bool) -> tuple[list[list[str]], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("format") != "OURS_VIT_HEAD_CALIBRATION_V1" or
            report.get("selection_fixed_before_private_inference") is not True):
        raise ValueError("assignment is not a head calibration report")
    assignment = report.get("assignment")
    provenance = "cost_aware_final"
    if assignment is None:
        if not allow_diagnostic:
            raise ValueError(
                "calibration has no final cost-aware assignment; "
                "use --allow-diagnostic only for oracle development")
        assignment = report["diagnostics"]["error_only_feasible"]["assignment"]
        provenance = "diagnostic_error_only_nonfinal"
    for layer in assignment:
        if any(path not in PATH_ORDER for path in layer):
            raise ValueError("assignment contains an unsupported protocol path")
    return assignment, provenance


def load_samples(manifest: Path, image_root: Path,
                 maximum: int) -> list[tuple[Path, int]]:
    result = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("manifest rows must be path,label")
            path = Path(row[0])
            result.append((path if path.is_absolute() else image_root / path,
                           int(row[1])))
            if len(result) == maximum:
                break
    if len(result) != maximum or any(not path.is_file() for path, _ in result):
        raise ValueError("insufficient readable oracle images")
    return result


def iter_batches(values: Any, size: int) -> Iterable[Any]:
    for start in range(0, values.shape[0], size):
        yield values[start:start + size]


def permutation_for(paths: list[str]) -> list[int]:
    # Python's sort is stable, retaining the logical order within a protocol.
    return sorted(range(len(paths)), key=lambda head: PATH_ORDER[paths[head]])


def feature_permutation(order: list[int], head_dim: int, torch: Any) -> Any:
    values = []
    for logical_head in order:
        values.extend(range(logical_head * head_dim,
                            (logical_head + 1) * head_dim))
    return torch.tensor(values, dtype=torch.long)


def reorder_model(model: Any, assignment: list[list[str]], torch: Any) -> list[list[int]]:
    layers = model.vit.encoder.layer
    heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // heads
    if len(assignment) != len(layers) or any(len(row) != heads for row in assignment):
        raise ValueError("assignment shape does not match checkpoint")
    orders = []
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            order = permutation_for(assignment[layer_index])
            indices = feature_permutation(order, head_dim, torch)
            attention = layer.attention.attention
            for projection in (attention.query, attention.key, attention.value):
                original_weight = projection.weight.detach().clone()
                projection.weight.copy_(original_weight.index_select(0, indices))
                if projection.bias is not None:
                    original_bias = projection.bias.detach().clone()
                    projection.bias.copy_(original_bias.index_select(0, indices))
            output = layer.attention.output.dense
            original_output = output.weight.detach().clone()
            output.weight.copy_(original_output.index_select(1, indices))
            orders.append(order)
    return orders


def capture_inputs(model: Any) -> tuple[list[Any], list[Any]]:
    values: list[Any] = [None] * len(model.vit.encoder.layer)
    handles = []
    for index, layer in enumerate(model.vit.encoder.layer):
        def hook(module: Any, inputs: tuple[Any, ...],
                 index: int = index) -> None:
            del module
            values[index] = inputs[0].detach()
        handles.append(layer.attention.attention.register_forward_pre_hook(hook))
    return values, handles


def close_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def attention_stages(layer: Any, hidden: Any, torch: Any,
                     fixed_approximate: bool) -> dict[str, Any]:
    attention = layer.attention.attention
    heads = attention.num_attention_heads
    head_dim = attention.attention_head_size

    def quantize(value: Any, scale: int) -> Any:
        if not fixed_approximate:
            return value
        factor = float(1 << scale)
        return torch.round(value * factor) / factor

    if fixed_approximate:
        hidden_int = torch.round(hidden * (1 << 12)).to(torch.int64)

        def fixed_linear(projection: Any, weight_scale: int,
                         output_scale: int) -> Any:
            weight_int = torch.round(
                projection.weight * (1 << weight_scale)).to(torch.int64)
            raw_scale = 12 + weight_scale
            raw = torch.matmul(hidden_int, weight_int.transpose(0, 1))
            if projection.bias is not None:
                raw = raw + torch.round(
                    projection.bias * (1 << raw_scale)).to(torch.int64)
            output = torch.floor_divide(
                raw, 1 << (raw_scale - output_scale))
            output = output.double() / float(1 << output_scale)
            return attention.transpose_for_scores(output)

        q = fixed_linear(attention.query, 10, 10)
        k = fixed_linear(attention.key, 10, 10)
        v = fixed_linear(attention.value, 11, 12)
    else:
        q = attention.transpose_for_scores(attention.query(hidden))
        k = attention.transpose_for_scores(attention.key(hidden))
        v = attention.transpose_for_scores(attention.value(hidden))
    qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    qk = quantize(qk, 12)
    probability = torch.softmax(qk, dim=-1)
    probability = quantize(probability, 12)
    context = torch.matmul(probability, v)
    context = quantize(context, 12)
    merged = context.permute(0, 2, 1, 3).contiguous().view(
        hidden.shape[0], hidden.shape[1], heads * head_dim)
    if fixed_approximate:
        # Exact integer accumulation makes the fixed-point permutation oracle
        # independent of floating GEMM reduction order.  Scale-12 context and
        # scale-10 WO weights produce scale-22 raw accumulators.
        merged_int = torch.round(merged * (1 << 12)).to(torch.int64)
        weight_int = torch.round(
            layer.attention.output.dense.weight * (1 << 10)).to(torch.int64)
        bias_int = torch.round(
            layer.attention.output.dense.bias * (1 << 22)).to(torch.int64)
        raw = torch.matmul(merged_int, weight_int.transpose(0, 1)) + bias_int
        wo = torch.floor_divide(raw, 1 << 10).double() / float(1 << 12)
    else:
        wo = torch.matmul(
            merged, layer.attention.output.dense.weight.transpose(0, 1))
        wo = wo + layer.attention.output.dense.bias
    return {"q": q, "k": k, "v": v, "qk": qk,
            "normalization": probability, "av": context, "wo": wo}


class Metrics:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float | int]] = {}

    def add(self, name: str, lhs: Any, rhs: Any) -> None:
        difference = (rhs - lhs).double()
        row = self.rows.setdefault(
            name, {"max_abs": 0.0, "sum_sq": 0.0, "elements": 0})
        row["max_abs"] = max(float(row["max_abs"]),
                             float(difference.abs().max().item()))
        row["sum_sq"] = float(row["sum_sq"]) + float(
            (difference * difference).sum().item())
        row["elements"] = int(row["elements"]) + difference.numel()

    def report(self, tolerance: float) -> dict[str, dict[str, float | int | str]]:
        result = {}
        for name, row in self.rows.items():
            mse = float(row["sum_sq"]) / int(row["elements"])
            result[name] = {
                "max_abs": float(row["max_abs"]),
                "mse": mse,
                "rmse": math.sqrt(mse),
                "elements": int(row["elements"]),
                "status": ("PASS" if float(row["max_abs"]) <= tolerance
                           else "FAIL"),
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("oracle sample and batch counts must be positive")

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, ViTForImageClassification
    except ImportError as error:
        raise RuntimeError("reorder oracle requires the prepared ViT environment") from error
    torch.set_grad_enabled(False)
    assignment, provenance = load_assignment(
        args.calibration_report, args.allow_diagnostic)
    if provenance != "cost_aware_final" and args.output_checkpoint is not None:
        raise ValueError("diagnostic assignments cannot be exported as checkpoints")

    original = ViTForImageClassification.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True, use_safetensors=True)
    reordered = ViTForImageClassification.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True, use_safetensors=True)
    original.eval()
    reordered.eval()
    orders = reorder_model(reordered, assignment, torch)

    samples = load_samples(args.manifest, args.image_root, args.samples)
    processor = AutoImageProcessor.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True)
    images = []
    for path, _ in samples:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    pixels = processor(images=images, return_tensors="pt").pixel_values
    fp32 = Metrics()
    fixed = Metrics()
    head_dim = original.config.hidden_size // original.config.num_attention_heads

    with torch.inference_mode():
        for batch in iter_batches(pixels, args.batch_size):
            original_inputs, original_handles = capture_inputs(original)
            reordered_inputs, reordered_handles = capture_inputs(reordered)
            try:
                original_logits = original(pixel_values=batch).logits
                reordered_logits = reordered(pixel_values=batch).logits
            finally:
                close_handles(original_handles)
                close_handles(reordered_handles)
            fp32.add("logits", original_logits, reordered_logits)
            fixed.add("logits", torch.round(original_logits * 4096) / 4096,
                      torch.round(reordered_logits * 4096) / 4096)
            for layer_index, order in enumerate(orders):
                if original_inputs[layer_index] is None or reordered_inputs[layer_index] is None:
                    raise RuntimeError("attention input hook did not execute")
                for is_fixed, metric in ((False, fp32), (True, fixed)):
                    before = attention_stages(
                        original.vit.encoder.layer[layer_index],
                        original_inputs[layer_index], torch, is_fixed)
                    # Use the same layer input to isolate the exact parameter
                    # permutation.  Whole-model logits above independently
                    # cover propagation through all reordered layers.
                    after = attention_stages(
                        reordered.vit.encoder.layer[layer_index],
                        original_inputs[layer_index], torch, is_fixed)
                    head_index = torch.tensor(order, dtype=torch.long)
                    for name in ("q", "k", "v", "qk", "normalization", "av"):
                        expected = before[name].index_select(1, head_index)
                        metric.add(f"layer_{layer_index}.{name}",
                                   expected, after[name])
                    metric.add(f"layer_{layer_index}.wo", before["wo"],
                               after["wo"])

    # Reordering output rows changes the GEMM output layout and may alter the
    # FP32 kernel's reduction/vectorization by a few ulps even though every
    # mathematical dot product is unchanged.  Fixed integer arithmetic below
    # remains bit-exact.
    fp32_tolerance = 2e-5
    fp32_report = fp32.report(fp32_tolerance)
    # Attention stages use exact integer accumulation and must be identical.
    # The final logits row is a scale-12 quantization of the two FP32 whole
    # model runs (not a second full fixed graph), so one output LSB is the
    # explicit acceptance bound for that row.
    fixed_tolerance = 1.0 / (1 << 12)
    fixed_report = fixed.report(fixed_tolerance)
    fp32_max = max(row["max_abs"] for row in fp32_report.values())
    fixed_max = max(row["max_abs"] for row in fixed_report.values())
    fixed_attention_exact = all(
        row["max_abs"] == 0.0 for name, row in fixed_report.items()
        if name != "logits")
    status = ("PASS" if fp32_max <= fp32_tolerance and
              fixed_max <= fixed_tolerance and fixed_attention_exact else
              "FAIL")
    if args.output_checkpoint is not None:
        args.output_checkpoint.mkdir(parents=True, exist_ok=False)
        reordered.save_pretrained(args.output_checkpoint, safe_serialization=True)
        processor.save_pretrained(args.output_checkpoint)

    mappings = []
    for layer, order in enumerate(orders):
        inverse = [0] * len(order)
        for physical, logical in enumerate(order):
            inverse[logical] = physical
        mappings.append({
            "layer": layer,
            "physical_to_logical": order,
            "logical_to_physical": inverse,
            "physical_protocols": [assignment[layer][logical]
                                   for logical in order],
        })
    report = {
        "format": "OURS_VIT_HEAD_REORDER_ORACLE_V1",
        "status": status,
        "assignment_provenance": provenance,
        "final_experiment_eligible": provenance == "cost_aware_final",
        "checkpoint_sha256": sha256_file(
            args.checkpoint_dir / "model.safetensors"),
        "calibration_report_sha256": sha256_file(args.calibration_report),
        "samples": args.samples,
        "parameter_convention": {
            "qkv": "permute output rows because HF Linear is [output,input]",
            "wo": "permute matching input columns; output rows and bias unchanged",
        },
        "mappings": mappings,
        "fp32": {"maximum_max_abs": fp32_max,
                 "acceptance_tolerance": fp32_tolerance,
                 "stages": fp32_report},
        "fixed_approximate": {"maximum_max_abs": fixed_max,
                              "logits_acceptance_tolerance": fixed_tolerance,
                              "attention_stages_require_exact": True,
                              "stages": fixed_report},
        "output_checkpoint": (str(args.output_checkpoint)
                              if args.output_checkpoint else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"HEAD_REORDER_STATUS={status}")
    print(f"HEAD_REORDER_ASSIGNMENT_PROVENANCE={provenance}")
    print(f"HEAD_REORDER_FP32_MAX_ABS={fp32_max:.17g}")
    print(f"HEAD_REORDER_FIXED_MAX_ABS={fixed_max:.17g}")
    print(f"HEAD_REORDER_REPORT={args.output}")
    if status != "PASS":
        raise SystemExit("head reorder oracle failed")


if __name__ == "__main__":
    main()
