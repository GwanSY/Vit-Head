#!/usr/bin/env python3
"""Run BLB's 12-layer fixed-point BERT reference on the MRPC assets.

This is a clear reference, not secure inference.  It intentionally reuses the
fixed-point arithmetic functions in ``bert.py`` while replacing that file's
machine-specific main program.  The function-only load also drops the unused
PyTorch/Transformers imports, so the reference needs only NumPy.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np


BLB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BLB_ROOT / "checkpoints" / "official-blb-mrpc" / "weights_txt"
PAPER_MRPC_ACC_AND_F1 = 88.5
EXPECTED_SAMPLES = 408
NUM_LAYERS = 12


def load_fixed_clear_functions():
    """Load only bert.py's arithmetic definitions, without its legacy CLI."""
    source_path = BLB_ROOT / "bert.py"
    source = source_path.read_text(encoding="utf-8")
    marker = 'if __name__ == "__main__":'
    if source.count(marker) != 1:
        raise RuntimeError("bert.py main marker changed; refusing an ambiguous load")
    definitions = source.split(marker, 1)[0]

    # These imports are used only by bert.py's legacy CLI (or commented code).
    # Keeping the arithmetic in one source of truth avoids copying the protocol
    # approximation while making this reference independent of PyTorch.
    transformers_import = (
        "from transformers import glue_compute_metrics as compute_metrics\n"
    )
    legacy_imports = "import warnings, argparse, tqdm, torch, copy\n"
    if definitions.count(transformers_import) != 1 or definitions.count(legacy_imports) != 1:
        raise RuntimeError(
            "bert.py imports changed; refusing to silently add reference dependencies"
        )
    definitions = definitions.replace(transformers_import, "")
    definitions = definitions.replace(legacy_imports, "import warnings, copy\n")
    module = types.ModuleType("blb_bert_fixed_clear_arithmetic")
    module.__file__ = str(source_path)
    exec(compile(definitions, str(source_path), "exec"), module.__dict__)
    return module


def load_txt(data_dir, name, dtype=None):
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError("required checkpoint file is missing: {}".format(path))
    value = np.loadtxt(str(path), delimiter=",")
    if dtype is not None:
        value = value.astype(dtype)
    return value


def load_weights(data_dir):
    names = {
        "wq": "attention.self.query.weight",
        "wk": "attention.self.key.weight",
        "wv": "attention.self.value.weight",
        "bq": "attention.self.query.bias",
        "bk": "attention.self.key.bias",
        "bv": "attention.self.value.bias",
        "w_att_o": "attention.output.dense.weight",
        "b_att_o": "attention.output.dense.bias",
        "w_inter": "intermediate.dense.weight",
        "b_inter": "intermediate.dense.bias",
        "w_out": "output.dense.weight",
        "b_out": "output.dense.bias",
        "w_ln_att": "attention.output.LayerNorm.weight",
        "b_ln_att": "attention.output.LayerNorm.bias",
        "w_ln_out": "output.LayerNorm.weight",
        "b_ln_out": "output.LayerNorm.bias",
    }
    weights = {name: [] for name in names}
    for layer in range(NUM_LAYERS):
        prefix = "bert.encoder.layer.{}.".format(layer)
        for key, suffix in names.items():
            value = load_txt(data_dir, prefix + suffix + ".txt")
            if key in ("wq", "wk", "wv"):
                value = value.astype(np.int64).reshape((12, 768, 64))
            elif key in ("bq", "bk", "bv"):
                value = value.astype(np.int64).reshape((12, 64))
            elif key not in ("w_ln_att", "b_ln_att", "w_ln_out", "b_ln_out"):
                value = value.astype(np.int64)
            weights[key].append(value)

    weights.update(
        {
            "w_pool": load_txt(
                data_dir, "bert.pooler.dense.weight.txt", np.int64
            ),
            "b_pool": load_txt(data_dir, "bert.pooler.dense.bias.txt", np.int64),
            "w_classifier": load_txt(data_dir, "classifier.weight.txt", np.int64),
            "b_classifier": load_txt(data_dir, "classifier.bias.txt", np.int64),
        }
    )
    return weights


def binary_mrpc_metrics(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    accuracy = float(np.mean(predictions == labels))
    true_positive = int(np.sum((predictions == 1) & (labels == 1)))
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    false_negative = int(np.sum((predictions == 0) & (labels == 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2.0 * true_positive / denominator) if denominator else 0.0
    return {
        "accuracy": accuracy,
        "f1": f1,
        "acc_and_f1": (accuracy + f1) / 2.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the real 12-layer BLB BERT fixed-clear MRPC reference"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the exported BLB MRPC weights and inputs",
    )
    parser.add_argument(
        "--samples",
        "--sample-num",
        dest="samples",
        type=int,
        default=EXPECTED_SAMPLES,
        help="Number of contiguous samples starting at inputs_0 (default: 408)",
    )
    parser.add_argument(
        "--paper-target",
        type=float,
        default=PAPER_MRPC_ACC_AND_F1,
        help="Paper Table 8 BLB MRPC acc_and_f1 percentage",
    )
    parser.add_argument(
        "--paper-tolerance-pp",
        type=float,
        default=0.1,
        help="Allowed absolute difference in percentage points (default: 0.1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples; 0 disables progress",
    )
    parser.add_argument(
        "--json-out", type=Path, help="Optionally write the final result as JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    labels = load_txt(data_dir, "labels.txt", np.int64).reshape(-1)
    if args.samples > labels.size:
        raise ValueError(
            "requested {} samples, but labels.txt contains {}".format(
                args.samples, labels.size
            )
        )

    arithmetic = load_fixed_clear_functions()
    load_started = time.perf_counter()
    weights = load_weights(data_dir)
    arithmetic.x_scale = load_txt(data_dir, "x_scale.txt")
    arithmetic.w_scale = load_txt(data_dir, "w_scale.txt")
    arithmetic.b_scale = load_txt(data_dir, "b_scale.txt")
    # bert.py shares these flags between its legacy CLI and arithmetic helpers.
    # This entry point reproduces the paper's dense (non-pruned) BERT-base path.
    arithmetic.args = types.SimpleNamespace(sparse=False, online_prune=False)
    for scale_name in ("x_scale", "w_scale", "b_scale"):
        if getattr(arithmetic, scale_name).shape != (4, NUM_LAYERS):
            raise ValueError("{} must have shape (4, 12)".format(scale_name))
    arithmetic.layer_norm_max = 0
    weight_load_seconds = time.perf_counter() - load_started

    predictions = []
    input_load_seconds = 0.0
    compute_seconds = 0.0
    inference_started = time.perf_counter()
    for sample in range(args.samples):
        input_started = time.perf_counter()
        hidden = load_txt(data_dir, "inputs_{}_data.txt".format(sample), np.int64)
        mask = load_txt(data_dir, "inputs_{}_mask.txt".format(sample), np.int64)
        input_load_seconds += time.perf_counter() - input_started
        if hidden.shape != (128, 768):
            raise ValueError("sample {} input must have shape (128, 768)".format(sample))
        if mask.shape != (128,):
            raise ValueError("sample {} mask must have shape (128,)".format(sample))

        compute_started = time.perf_counter()
        for layer in range(NUM_LAYERS):
            hidden = arithmetic.attention_layer(
                hidden,
                weights["wq"][layer],
                weights["wk"][layer],
                weights["wv"][layer],
                weights["bq"][layer],
                weights["bk"][layer],
                weights["bv"][layer],
                weights["w_att_o"][layer],
                weights["b_att_o"][layer],
                weights["w_inter"][layer],
                weights["b_inter"][layer],
                weights["w_out"][layer],
                weights["b_out"][layer],
                weights["w_ln_att"][layer],
                weights["b_ln_att"][layer],
                weights["w_ln_out"][layer],
                weights["b_ln_out"][layer],
                mask,
                layer,
            )
        logits = arithmetic.pool_classify_layer(
            hidden[0],
            weights["w_pool"],
            weights["w_classifier"],
            weights["b_pool"],
            weights["b_classifier"],
        )
        predictions.append(int(np.argmax(logits)))
        compute_seconds += time.perf_counter() - compute_started

        completed = sample + 1
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == args.samples
        ):
            partial = binary_mrpc_metrics(predictions, labels[:completed])
            print(
                "BERT_FIXED_CLEAR_PROGRESS samples={}/{} acc={:.6f} f1={:.6f} "
                "acc_and_f1={:.6f}".format(
                    completed,
                    args.samples,
                    partial["accuracy"],
                    partial["f1"],
                    partial["acc_and_f1"],
                ),
                flush=True,
            )

    inference_seconds = time.perf_counter() - inference_started
    metrics = binary_mrpc_metrics(predictions, labels[: args.samples])
    observed_percent = metrics["acc_and_f1"] * 100.0
    signed_delta_pp = observed_percent - args.paper_target
    full_dataset = args.samples == EXPECTED_SAMPLES and labels.size == EXPECTED_SAMPLES
    if full_dataset:
        paper_status = (
            "PASS"
            if abs(signed_delta_pp) <= args.paper_tolerance_pp
            else "FAIL"
        )
    else:
        paper_status = "NOT_EVALUATED_PARTIAL_DATASET"

    result = {
        "mode": "fixed_clear_reference",
        "secure_inference": False,
        "task": "mrpc",
        "layers": NUM_LAYERS,
        "samples": args.samples,
        "dataset_samples": int(labels.size),
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "acc_and_f1": metrics["acc_and_f1"],
        "paper_metric": "mrpc_acc_and_f1_percent",
        "paper_target_percent": args.paper_target,
        "observed_percent": observed_percent,
        "signed_delta_percentage_points": signed_delta_pp,
        "absolute_delta_percentage_points": abs(signed_delta_pp),
        "paper_tolerance_percentage_points": args.paper_tolerance_pp,
        "paper_comparison_status": paper_status,
        "weight_load_seconds": weight_load_seconds,
        "input_load_seconds": input_load_seconds,
        "compute_seconds": compute_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "layer_norm_max": float(arithmetic.layer_norm_max),
        "data_dir": str(data_dir),
    }
    encoded = json.dumps(result, sort_keys=True)
    print("BERT_FIXED_CLEAR_RESULT {}".format(encoded))
    print(
        "REFERENCE_ONLY secure_inference=no; this result does not validate the "
        "two-party CPU/GPU protocol"
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0 if paper_status != "FAIL" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print("BERT_FIXED_CLEAR_ERROR {}".format(error), file=sys.stderr)
        sys.exit(2)
