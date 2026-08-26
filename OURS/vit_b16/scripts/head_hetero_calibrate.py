#!/usr/bin/env python3
"""Offline, no-training calibration for heterogeneous ViT attention heads.

The script never calls backward and never mutates model parameters.  It fits
the public HE-linear coefficient alpha by least squares on public calibration
images, measures ReLU-Norm and HE-linear errors per layer/head, and evaluates
fixed (input-independent) assignments on logits and labels.  A final protocol
assignment is emitted only when a real Phantom CKKS cost profile is supplied;
otherwise the report is deliberately marked BLOCKED_COST_PROFILE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PATH_SOFTMAX = "softmax_mpc"
PATH_RELU = "relu_norm_mpc"
PATH_LINEAR = "he_linear"
PATHS = (PATH_SOFTMAX, PATH_RELU, PATH_LINEAR)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(manifest: Path, image_root: Path,
                 maximum: int) -> list[tuple[Path, int]]:
    result: list[tuple[Path, int]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("calibration manifest rows must be path,label")
            path = Path(row[0])
            path = path if path.is_absolute() else image_root / path
            result.append((path, int(row[1])))
            if maximum > 0 and len(result) == maximum:
                break
    if not result:
        raise ValueError("calibration manifest is empty")
    missing = next((path for path, _ in result if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return result


def iter_batches(pixel_values: Any, labels: Any,
                 batch_size: int) -> Iterable[tuple[Any, Any]]:
    for start in range(0, pixel_values.shape[0], batch_size):
        yield (pixel_values[start:start + batch_size],
               labels[start:start + batch_size])


@dataclass
class LeastSquares:
    zy: float = 0.0
    zz: float = 0.0


class AlphaCollector:
    def __init__(self, model: Any, torch: Any):
        self.model = model
        self.torch = torch
        layers = len(model.vit.encoder.layer)
        heads = model.config.num_attention_heads
        self.values = [[LeastSquares() for _ in range(heads)]
                       for _ in range(layers)]
        self.handles: list[Any] = []

    def install(self) -> None:
        for layer_index, layer in enumerate(self.model.vit.encoder.layer):
            attention = layer.attention.attention

            def hook(module: Any, inputs: tuple[Any, ...], output: Any,
                     layer_index: int = layer_index) -> None:
                del output
                hidden = inputs[0]
                q = module.transpose_for_scores(module.query(hidden))
                k = module.transpose_for_scores(module.key(hidden))
                v = module.transpose_for_scores(module.value(hidden))
                score = self.torch.matmul(q, k.transpose(-1, -2))
                score = score / math.sqrt(module.attention_head_size)
                target = self.torch.matmul(
                    self.torch.softmax(score, dim=-1), v)
                unscaled = self.torch.matmul(score, v)
                for head in range(module.num_attention_heads):
                    row = self.values[layer_index][head]
                    z = unscaled[:, head].double()
                    y = target[:, head].double()
                    row.zy += float((z * y).sum().item())
                    row.zz += float((z * z).sum().item())

            self.handles.append(attention.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def alphas(self) -> list[list[float]]:
        result: list[list[float]] = []
        for layer in self.values:
            result.append([
                row.zy / row.zz if row.zz > 0.0 else 0.0
                for row in layer
            ])
        return result


class ErrorCollector:
    def __init__(self, model: Any, torch: Any,
                 alphas: list[list[float]]):
        self.model = model
        self.torch = torch
        self.alphas = alphas
        layers = len(model.vit.encoder.layer)
        heads = model.config.num_attention_heads
        self.rows: list[list[dict[str, Any]]] = [[{
            "target_context_sq": 0.0,
            "target_projected_sq": 0.0,
            PATH_RELU: {"context_sq": 0.0, "projected_sq": 0.0,
                        "context_max_abs": 0.0,
                        "projected_max_abs": 0.0},
            PATH_LINEAR: {"context_sq": 0.0, "projected_sq": 0.0,
                          "context_max_abs": 0.0,
                          "projected_max_abs": 0.0},
            "context_elements": 0,
            "projected_elements": 0,
        } for _ in range(heads)] for _ in range(layers)]
        self.handles: list[Any] = []

    def install(self) -> None:
        for layer_index, layer in enumerate(self.model.vit.encoder.layer):
            attention = layer.attention.attention
            output_weight = layer.attention.output.dense.weight

            def hook(module: Any, inputs: tuple[Any, ...], output: Any,
                     layer_index: int = layer_index,
                     output_weight: Any = output_weight) -> None:
                del output
                hidden = inputs[0]
                q = module.transpose_for_scores(module.query(hidden))
                k = module.transpose_for_scores(module.key(hidden))
                v = module.transpose_for_scores(module.value(hidden))
                score = self.torch.matmul(q, k.transpose(-1, -2))
                score = score / math.sqrt(module.attention_head_size)
                softmax_context = self.torch.matmul(
                    self.torch.softmax(score, dim=-1), v)
                relu_score = self.torch.relu(score)
                relu_probability = relu_score / (
                    relu_score.sum(dim=-1, keepdim=True) + 1e-12)
                relu_context = self.torch.matmul(relu_probability, v)
                linear_context = self.torch.matmul(score, v)
                for head in range(module.num_attention_heads):
                    start = head * module.attention_head_size
                    end = start + module.attention_head_size
                    target = softmax_context[:, head]
                    candidates = {
                        PATH_RELU: relu_context[:, head],
                        PATH_LINEAR: (linear_context[:, head] *
                                      self.alphas[layer_index][head]),
                    }
                    target_projected = self.torch.matmul(
                        target, output_weight[:, start:end].transpose(0, 1))
                    row = self.rows[layer_index][head]
                    row["target_context_sq"] += float(
                        (target.double() ** 2).sum().item())
                    row["target_projected_sq"] += float(
                        (target_projected.double() ** 2).sum().item())
                    row["context_elements"] += target.numel()
                    row["projected_elements"] += target_projected.numel()
                    for path, candidate in candidates.items():
                        difference = candidate - target
                        projected = self.torch.matmul(
                            difference,
                            output_weight[:, start:end].transpose(0, 1))
                        metric = row[path]
                        metric["context_sq"] += float(
                            (difference.double() ** 2).sum().item())
                        metric["projected_sq"] += float(
                            (projected.double() ** 2).sum().item())
                        metric["context_max_abs"] = max(
                            metric["context_max_abs"],
                            float(difference.abs().max().item()))
                        metric["projected_max_abs"] = max(
                            metric["projected_max_abs"],
                            float(projected.abs().max().item()))

            self.handles.append(attention.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self) -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for layer_index, layer in enumerate(self.rows):
            layer_rows: list[dict[str, Any]] = []
            for head, row in enumerate(layer):
                item: dict[str, Any] = {
                    "layer": layer_index,
                    "head": head,
                    "alpha": self.alphas[layer_index][head],
                    "candidates": {},
                }
                for path in (PATH_RELU, PATH_LINEAR):
                    metric = row[path]
                    context_mse = metric["context_sq"] / row["context_elements"]
                    projected_mse = (
                        metric["projected_sq"] / row["projected_elements"])
                    item["candidates"][path] = {
                        "context_max_abs": metric["context_max_abs"],
                        "context_mse": context_mse,
                        "context_nrmse": math.sqrt(
                            metric["context_sq"] /
                            max(row["target_context_sq"], 1e-30)),
                        "projected_max_abs": metric["projected_max_abs"],
                        "projected_mse": projected_mse,
                        "projected_nrmse": math.sqrt(
                            metric["projected_sq"] /
                            max(row["target_projected_sq"], 1e-30)),
                    }
                layer_rows.append(item)
            result.append(layer_rows)
        return result


def install_fixed_assignment(model: Any, torch: Any,
                             assignment: list[list[str]],
                             alphas: list[list[float]]) -> list[tuple[Any, Any]]:
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(model.vit.encoder.layer):
        module = layer.attention.attention
        originals.append((module, module.forward))

        def forward(self: Any, hidden_states: Any, head_mask: Any = None,
                    output_attentions: bool = False,
                    layer_index: int = layer_index) -> tuple[Any, ...]:
            q = self.transpose_for_scores(self.query(hidden_states))
            k = self.transpose_for_scores(self.key(hidden_states))
            v = self.transpose_for_scores(self.value(hidden_states))
            score = torch.matmul(q, k.transpose(-1, -2))
            score = score / math.sqrt(self.attention_head_size)
            probability = torch.softmax(score, dim=-1)
            rows = []
            for head, path in enumerate(assignment[layer_index]):
                if path == PATH_SOFTMAX:
                    row = probability[:, head]
                elif path == PATH_RELU:
                    relu = torch.relu(score[:, head])
                    row = relu / (relu.sum(dim=-1, keepdim=True) + 1e-12)
                elif path == PATH_LINEAR:
                    row = score[:, head] * alphas[layer_index][head]
                else:
                    raise ValueError(f"unsupported attention path: {path}")
                rows.append(row)
            selected = torch.stack(rows, dim=1)
            selected = self.dropout(selected)
            if head_mask is not None:
                selected = selected * head_mask
            context = torch.matmul(selected, v)
            context = context.permute(0, 2, 1, 3).contiguous()
            shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(shape)
            return ((context, selected) if output_attentions else (context,))

        module.forward = types.MethodType(forward, module)
    return originals


def restore_forwards(originals: list[tuple[Any, Any]]) -> None:
    for module, forward in originals:
        module.forward = forward


def evaluate(model: Any, pixel_values: Any, labels: Any, batch_size: int,
             torch: Any, baseline_logits: Any | None = None) -> tuple[dict[str, Any], Any]:
    logits_rows = []
    correct = 0
    for pixels, target in iter_batches(pixel_values, labels, batch_size):
        logits = model(pixel_values=pixels).logits
        logits_rows.append(logits.cpu())
        correct += int((logits.argmax(dim=-1) == target).sum().item())
    logits = torch.cat(logits_rows, dim=0)
    report: dict[str, Any] = {
        "samples": int(labels.numel()),
        "top1_correct": correct,
        "top1": correct / int(labels.numel()),
    }
    if baseline_logits is not None:
        difference = logits - baseline_logits
        report.update({
            "baseline_top1_consistency": float(
                (logits.argmax(dim=-1) ==
                 baseline_logits.argmax(dim=-1)).float().mean().item()),
            "logits_max_abs": float(difference.abs().max().item()),
            "logits_mse": float((difference.double() ** 2).mean().item()),
        })
    return report, logits


def validate_cost_profile(path: Path, layers: int, heads: int,
                          tokens: int) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (profile.get("format") != "OURS_VIT_HEAD_COST_V1" or
            profile.get("provenance") != "measured_secure_run" or
            profile.get("scheme") != "ckks" or
            profile.get("backend") != "phantom_gpu" or
            int(profile.get("tokens", 0)) != tokens or
            int(profile.get("layers", 0)) != layers or
            int(profile.get("heads", 0)) != heads):
        raise ValueError(
            "cost profile must be a measured Phantom-GPU CKKS ViT profile")
    paths = profile.get("paths", {})
    for name in PATHS:
        row = paths.get(name, {})
        required = ("batch_capacity_heads", "batch_latency_ms",
                    "batch_communication_bytes", "ciphertexts_per_batch",
                    "he_depth", "rotations_per_batch",
                    "ckks_to_mpc_per_batch", "mpc_to_ckks_per_batch",
                    "mpc_elements_per_head")
        if any(key not in row for key in required):
            raise ValueError(f"cost profile path is incomplete: {name}")
        if (int(row["batch_capacity_heads"]) <= 0 or
                float(row["batch_latency_ms"]) < 0.0 or
                int(row["batch_communication_bytes"]) < 0):
            raise ValueError(f"cost profile path is invalid: {name}")
    return profile


def layer_cost(counts: dict[str, int], profile: dict[str, Any]) -> dict[str, Any]:
    result = {
        "latency_ms": 0.0, "communication_bytes": 0,
        "ciphertexts": 0, "rotations": 0, "ckks_to_mpc": 0,
        "mpc_to_ckks": 0, "mpc_elements": 0, "max_he_depth": 0,
        "groups": {}, "packing_utilization": {},
    }
    for path in PATHS:
        count = counts[path]
        row = profile["paths"][path]
        capacity = int(row["batch_capacity_heads"])
        groups = math.ceil(count / capacity) if count else 0
        result["groups"][path] = groups
        result["packing_utilization"][path] = (
            count / (groups * capacity) if groups else 0.0)
        result["latency_ms"] += groups * float(row["batch_latency_ms"])
        result["communication_bytes"] += groups * int(
            row["batch_communication_bytes"])
        result["ciphertexts"] += groups * int(row["ciphertexts_per_batch"])
        result["rotations"] += groups * int(row["rotations_per_batch"])
        result["ckks_to_mpc"] += groups * int(
            row["ckks_to_mpc_per_batch"])
        result["mpc_to_ckks"] += groups * int(
            row["mpc_to_ckks_per_batch"])
        result["mpc_elements"] += count * int(row["mpc_elements_per_head"])
        if count:
            result["max_he_depth"] = max(
                result["max_he_depth"], int(row["he_depth"]))
    return result


def select_assignment(metrics: list[list[dict[str, Any]]],
                      profile: dict[str, Any],
                      maximum_nrmse: float) -> tuple[list[list[str]], list[dict[str, Any]]]:
    assignment: list[list[str]] = []
    costs: list[dict[str, Any]] = []
    for layer_rows in metrics:
        # Dynamic programming keeps the lowest-error feasible assignment for
        # every (softmax,relu,linear) count tuple.  The measured group cost is
        # then evaluated once per tuple, so packing is not reduced to a
        # misleading independent per-head cost.
        states: dict[tuple[int, int, int], tuple[float, list[str]]] = {
            (0, 0, 0): (0.0, [])}
        for row in layer_rows:
            eligible = [PATH_SOFTMAX]
            for path in (PATH_RELU, PATH_LINEAR):
                if row["candidates"][path]["projected_nrmse"] <= maximum_nrmse:
                    eligible.append(path)
            updated: dict[tuple[int, int, int], tuple[float, list[str]]] = {}
            for counts, (penalty, paths) in states.items():
                for path in eligible:
                    next_counts = list(counts)
                    next_counts[PATHS.index(path)] += 1
                    key = tuple(next_counts)
                    path_penalty = (0.0 if path == PATH_SOFTMAX else
                                    row["candidates"][path]["projected_mse"])
                    candidate = (penalty + path_penalty, paths + [path])
                    if key not in updated or candidate[0] < updated[key][0]:
                        updated[key] = candidate
            states = updated
        best: tuple[float, float, list[str], dict[str, Any]] | None = None
        for counts, (penalty, paths) in states.items():
            count_map = {path: counts[index] for index, path in enumerate(PATHS)}
            cost = layer_cost(count_map, profile)
            candidate = (cost["latency_ms"], penalty, paths, cost)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("no feasible fixed head assignment")
        assignment.append(best[2])
        costs.append(best[3])
    return assignment, costs


def fixed_assignment_evaluation(model: Any, torch: Any,
                                assignment: list[list[str]],
                                alphas: list[list[float]], pixel_values: Any,
                                labels: Any, batch_size: int,
                                baseline_logits: Any) -> dict[str, Any]:
    originals = install_fixed_assignment(model, torch, assignment, alphas)
    try:
        report, _ = evaluate(model, pixel_values, labels, batch_size, torch,
                             baseline_logits)
        return report
    finally:
        restore_forwards(originals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-profile", type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-projected-nrmse", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0:
        raise ValueError("calibration sample and batch counts must be positive")
    if not 0.0 <= args.max_projected_nrmse <= 1.0:
        raise ValueError("--max-projected-nrmse must be in [0,1]")

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, ViTForImageClassification
    except ImportError as error:
        raise RuntimeError("calibration requires the prepared ViT environment") from error

    torch.set_grad_enabled(False)
    samples = load_samples(args.manifest, args.image_root, args.max_samples)
    processor = AutoImageProcessor.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True)
    model = ViTForImageClassification.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True, use_safetensors=True)
    model.eval()
    images = []
    for path, _ in samples:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)

    alpha_collector = AlphaCollector(model, torch)
    alpha_collector.install()
    try:
        baseline, baseline_logits = evaluate(
            model, pixel_values, labels, args.batch_size, torch)
    finally:
        alpha_collector.close()
    alphas = alpha_collector.alphas()

    error_collector = ErrorCollector(model, torch, alphas)
    error_collector.install()
    try:
        evaluate(model, pixel_values, labels, args.batch_size, torch)
    finally:
        error_collector.close()
    metrics = error_collector.report()

    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    diagnostics: dict[str, Any] = {}
    for name, path in (("all_relu", PATH_RELU),
                       ("all_linear", PATH_LINEAR)):
        fixed = [[path] * heads for _ in range(layers)]
        diagnostics[name] = fixed_assignment_evaluation(
            model, torch, fixed, alphas, pixel_values, labels,
            args.batch_size, baseline_logits)
    # This is an accuracy diagnostic only, never the final protocol choice:
    # it ignores secure cost and picks the lower-error eligible approximation.
    # Keeping it separately named prevents an error-only rule from being
    # mistaken for the required cost-aware assignment.
    error_only_assignment: list[list[str]] = []
    for layer_rows in metrics:
        layer_assignment: list[str] = []
        for row in layer_rows:
            eligible = [
                path for path in (PATH_RELU, PATH_LINEAR)
                if row["candidates"][path]["projected_nrmse"] <=
                args.max_projected_nrmse
            ]
            layer_assignment.append(
                min(eligible,
                    key=lambda path: row["candidates"][path]["projected_mse"])
                if eligible else PATH_SOFTMAX)
        error_only_assignment.append(layer_assignment)
    diagnostics["error_only_feasible"] = {
        **fixed_assignment_evaluation(
            model, torch, error_only_assignment, alphas, pixel_values, labels,
            args.batch_size, baseline_logits),
        "assignment": error_only_assignment,
        "not_a_final_protocol_selection": True,
        "reason": "secure cost profile is not part of this diagnostic",
    }

    final_status = "BLOCKED_COST_PROFILE"
    assignment = None
    costs = None
    selected_evaluation = None
    profile_digest = None
    if args.cost_profile is not None:
        expected_tokens = (
            (pixel_values.shape[-1] // model.config.patch_size) ** 2 + 1)
        profile = validate_cost_profile(
            args.cost_profile, layers, heads, expected_tokens)
        if int(profile["tokens"]) != expected_tokens:
            raise ValueError("cost profile token count does not match checkpoint")
        assignment, costs = select_assignment(
            metrics, profile, args.max_projected_nrmse)
        selected_evaluation = fixed_assignment_evaluation(
            model, torch, assignment, alphas, pixel_values, labels,
            args.batch_size, baseline_logits)
        final_status = "PASS"
        profile_digest = sha256_file(args.cost_profile)

    gradients_present = any(parameter.grad is not None
                            for parameter in model.parameters())
    if gradients_present:
        raise RuntimeError("no-training calibration unexpectedly produced gradients")
    checkpoint_file = args.checkpoint_dir / "model.safetensors"
    report = {
        "format": "OURS_VIT_HEAD_CALIBRATION_V1",
        "status": final_status,
        "selection_fixed_before_private_inference": True,
        "private_input_dynamic_head_branching": False,
        "training": {"backward_calls": 0, "optimizer_steps": 0,
                     "gradients_present": False, "model_parameter_updates": 0},
        "checkpoint": {"directory": str(args.checkpoint_dir),
                       "model_sha256": sha256_file(checkpoint_file)},
        "calibration": {"manifest": str(args.manifest),
                        "manifest_sha256": sha256_file(args.manifest),
                        "samples": len(samples),
                        "sample_paths": [str(path) for path, _ in samples],
                        "public_data_only": True},
        "model": {"layers": layers, "heads": heads,
                  "head_dim": model.config.hidden_size // heads,
                  "tokens": ((pixel_values.shape[-1] //
                              model.config.patch_size) ** 2 + 1)},
        "thresholds": {"max_projected_nrmse": args.max_projected_nrmse},
        "baseline": baseline,
        "diagnostics": diagnostics,
        "heads": metrics,
        "cost_profile_sha256": profile_digest,
        "assignment": assignment,
        "layer_costs": costs,
        "selected_evaluation": selected_evaluation,
        "blocked_reason": (None if final_status == "PASS" else
                           "real measured Phantom-GPU CKKS head cost profile not supplied"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"HEAD_HETERO_CALIBRATION_STATUS={final_status}")
    print(f"HEAD_HETERO_CALIBRATION_SAMPLES={len(samples)}")
    print(f"HEAD_HETERO_BASELINE_TOP1={baseline['top1']:.8f}")
    print("HEAD_HETERO_ALL_RELU_TOP1="
          f"{diagnostics['all_relu']['top1']:.8f}")
    print("HEAD_HETERO_ALL_LINEAR_TOP1="
          f"{diagnostics['all_linear']['top1']:.8f}")
    print("HEAD_HETERO_ERROR_ONLY_TOP1="
          f"{diagnostics['error_only_feasible']['top1']:.8f}")
    print(f"HEAD_HETERO_CALIBRATION_REPORT={args.output}")


if __name__ == "__main__":
    main()
