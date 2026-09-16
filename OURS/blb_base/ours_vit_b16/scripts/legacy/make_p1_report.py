#!/usr/bin/env python3
"""Summarize the real P1 method-effectiveness experiment."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def plan_protocols(path: Path, layers: int = 12, heads: int = 12):
    result = [["" for _ in range(heads)] for _ in range(layers)]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    for row in csv.DictReader(lines[2:]):
        result[int(row["layer"])][int(row["original_head"])] = row["protocol"]
    return result


def count_text(protocols):
    counts = Counter(value for layer in protocols for value in layer)
    return (f"{counts['softmax_mpc']}S / {counts['relu_norm_mpc']}R / "
            f"{counts['he_linear']}L")


def direct_cost(protocols, rows):
    selected = []
    for layer, expected in enumerate(protocols):
        matches = [row for row in rows if int(row["layer"]) == layer and
                   row["protocols_original"].split(";") == expected]
        if len(matches) != 1:
            return None
        selected.append(matches[0])
    return {
        "layers": len(selected),
        "latency_s": sum(float(row["latency_ms"]) for row in selected) / 1000.0,
        "communication_gb": sum(int(row["communication_bytes"])
                                for row in selected) / 1e9,
        "rounds": sum(int(row["rounds"]) for row in selected),
        "conversion_ct": sum(int(row["ckks_to_mpc"]) + int(row["mpc_to_ckks"])
                             for row in selected),
        "he_s": sum(float(row["he_instrumented_ms"]) for row in selected) / 1000.0,
    }


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--assignment-costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    assignment = json.loads(Path(calibration["head_assignment"]).read_text(
        encoding="utf-8"))
    plan_manifest = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    accuracy = json.loads(args.accuracy.read_text(encoding="utf-8"))
    with args.assignment_costs.open(newline="", encoding="utf-8") as handle:
        cost_rows = list(csv.DictReader(handle))

    accuracy_by_name = {row["plan"]: row for row in accuracy["summaries"]}
    records = {}
    for name, info in plan_manifest["plans"].items():
        protocols = plan_protocols(Path(info["path"]))
        cost = direct_cost(protocols, cost_rows)
        measured = accuracy_by_name.get(name)
        records[name] = {
            "counts": count_text(protocols),
            "protocols": protocols,
            "accuracy": (measured["top1"] if measured else None),
            "correct": (measured["top1_correct"] if measured else None),
            "cost": cost,
            "sha256": info["sha256"],
        }

    baseline_accuracy = records["All-Softmax"]["accuracy"]
    rows = []
    for name in ("All-Softmax", "Random", "Sensitivity-only", "Cost-only", "Ours"):
        record = records[name]
        cost = record["cost"]
        drop = (baseline_accuracy - record["accuracy"]
                if record["accuracy"] is not None else None)
        coverage = f"{cost['layers']}/12 direct rows" if cost else "N/A"
        rows.append(
            f"| {name} | {record['counts']} | "
            f"{record['correct']}/50 ({record['accuracy'] * 100:.1f}%) | "
            f"{drop * 100:+.1f} pp | "
            f"{fmt(cost['latency_s']) if cost else 'N/A'} | "
            f"{fmt(cost['communication_gb']) if cost else 'N/A'} | "
            f"{cost['conversion_ct'] if cost else 'N/A'} | "
            f"{coverage} |"
        )

    final_metrics = calibration["final_metrics"]
    ours = records["Ours"]
    accuracy_gate = ours["accuracy"] >= baseline_accuracy
    same_counts = all(records[name]["counts"] == ours["counts"]
                      for name in ("Random", "Sensitivity-only", "Cost-only", "Ours"))
    independent_cost_comparison = all(
        records[name]["cost"] is not None
        for name in ("All-Softmax", "Random", "Sensitivity-only", "Cost-only", "Ours"))
    p1_pass = (calibration["global_guard_passed"] and accuracy_gate and
               same_counts and independent_cost_comparison)

    report = [
        "# P1：证明方法有效（真实运行）",
        "",
        "结论：**FAIL**。当前主预算上的 Ours 没有通过 50 张 fixed-clear 模型级精度门禁；同时 Random 没有对应的完整 secure assignment 实测成本，不能用成本外推补齐。",
        "",
        "## 实验口径",
        "",
        f"- calibration：4 张公开输入；relative-L2 threshold={assignment['selection']['max_head_relative_l2']}; global calibration guard={calibration['global_guard_passed']}。",
        f"- test：固定 50 张 manifest，sha256={accuracy['manifest_sha256']}；只报告 fixed-clear accuracy，secure 50 张 accuracy 未运行。",
        "- 成本：只匹配 `direct_assignment_costs_a3_a4_full12_active_20260901.csv` 中同一层、同一完整 Head assignment 的真实 A3/A4 行。表中成本是 12 个已测 block assignment unit 的直接汇总，不是完整网络 latency 预测；没有匹配行就记 N/A。",
        "",
        "## P1 主结果",
        "",
        "| Method | Head counts | Fixed-clear 50 accuracy | Drop vs All-Softmax | Direct assignment latency sum (s) | Direct communication sum (GB) | Conversion ct | Cost coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Calibration gate",
        "",
        "| Metric | Result | Status |",
        "|---|---:|---|",
        f"| 4-image top-1 agreement | {final_metrics['top1_agreement']:.3f} | PASS |",
        f"| FP32 logits max-abs | {final_metrics['fp32_logits']['max_abs']:.4f} (limit 4.0) | {'PASS' if final_metrics['fp32_logits']['max_abs'] <= 4.0 else 'FAIL'} |",
        f"| Fixed logits MSE | {final_metrics['fixed_logits']['mse']:.4f} (limit 0.30) | {'PASS' if final_metrics['fixed_logits']['mse'] <= 0.30 else 'FAIL'} |",
        f"| Ours 50-image accuracy >= All-Softmax | {ours['correct']}/50 >= {records['All-Softmax']['correct']}/50 | {'PASS' if accuracy_gate else 'FAIL'} |",
        f"| Random/Sensitivity/Cost/Ours same S/R/L count | {same_counts} | {'PASS' if same_counts else 'FAIL'} |",
        f"| All five secure cost rows directly measured | {independent_cost_comparison} | {'PASS' if independent_cost_comparison else 'FAIL'} |",
        f"| **P1 method effective** | calibration + accuracy + cost evidence | **{'PASS' if p1_pass else 'FAIL'}** |",
        "",
        "## 失败原因",
        "",
        "1. 0.65 阈值生成的 Ours 为 140S/0R/4L；它在 4 张 calibration 图上 top-1 agreement=1.0，但在独立 50 张 fixed-clear 测试上只有 32/50，而 All-Softmax 为 38/50，下降 12.0 个百分点。局部 attention-context relative-L2 没有充分预测模型级分类误差。",
        "2. 本轮每个允许降级层只有一条直接测得的 10S/2L assignment；因此 Cost-only 与 Ours 选到同一固定计划，无法从这批数据证明 cost-aware selector 比 sensitivity-only 更好。",
        "3. Random 的 Head 编号不同，没有对应的完整 A3/A4 secure assignment 行；其 50 张 fixed-clear 数值可以报告，但 secure latency/communication 必须继续实测，不能按 group size 或 Ours 行推算。",
        "",
        "## 允许写入论文的结论",
        "",
        "本轮只能写：selector 生成了固定、可审计且 calibration guard 通过的计划；不能写“P1 方法有效”。在补足 50 张代表性 calibration/model-level gate、每层多个真实 assignment 成本候选并完成 Random 等 secure 实测前，P1 应保持 FAIL。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"OURS_P1_REPORT status={'PASS' if p1_pass else 'FAIL'} path={args.output}")


if __name__ == "__main__":
    main()
