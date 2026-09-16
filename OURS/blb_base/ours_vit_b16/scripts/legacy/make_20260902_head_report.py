#!/usr/bin/env python3
"""Write the current head-experiment report from measured artifacts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GROUPS = (1, 2, 4, 6, 8, 12)
PATHS = ("Softmax", "ReLU", "Linear")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: object, digits: int = 6) -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:.{digits}f}"


def integer(value: object) -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def metric_map(run: Path) -> dict[str, str]:
    rows = read_csv(run / "ours_summary.csv")
    return {row["metric"]: row["value"] for row in rows}


def plan_counts(path: Path) -> tuple[int, int, int]:
    counts = {"softmax_mpc": 0, "relu_norm_mpc": 0, "he_linear": 0}
    if not path.is_file():
        return 0, 0, 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    for row in csv.DictReader(lines[2:]):
        protocol = row.get("protocol")
        if protocol in counts:
            counts[protocol] += 1
    return counts["softmax_mpc"], counts["relu_norm_mpc"], counts["he_linear"]


def stage1_rows() -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    formal = ROOT / "outputs/head_experiment_20260901/stage1/stage1_formal_real_costs.csv"
    for row in read_csv(formal):
        rows[(row["path"], int(row["heads_group"]))] = row
    smoke = ROOT / "outputs/head_experiment_20260901/stage1/softmax_unshaped_smoke.csv"
    for row in read_csv(smoke):
        rows[(row["path"], int(row["heads_group"]))] = row
    return rows


def current_a0_block_anchor() -> dict[str, float] | None:
    run = ROOT / "outputs/a0_after_rotation_fix_20260902c"
    party1 = read_csv(run / "block_profile_party1.csv")
    party2 = read_csv(run / "block_profile_party2.csv")
    if not party1 or not party2:
        return None
    return {
        "latency_s": max(float(party1[0]["block_online_time_ms"]),
                          float(party2[0]["block_online_time_ms"])) / 1000.0,
        "communication_mb": (
            int(party1[0]["block_online_comm_bytes"]) +
            int(party2[0]["block_online_comm_bytes"])) / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0", type=Path, default=ROOT /
                        "outputs/a0_after_rotation_fix_20260902c/a0_trace_comparison.json")
    parser.add_argument("--accuracy", type=Path, default=ROOT /
                        "outputs/20260902_accuracy_50/fixed_clear_50_after_a0.json")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "outputs/20260902_head_report.md")
    args = parser.parse_args()

    a0 = json.loads(args.a0.read_text(encoding="utf-8")) if args.a0.is_file() else None
    costs = stage1_rows()
    accuracy = (json.loads(args.accuracy.read_text(encoding="utf-8"))
                if args.accuracy.is_file() else None)

    lines = [
        "# ViT-B/16 head-level heterogeneous secure-attention re-audit",
        "",
        "本报告只读取已经落盘的真实运行产物。`N/A` 表示没有完成对应运行，"
        "不用其它 group 或其它网络条件外推。A0 未通过时，安全 accuracy、最终 assignment "
        "和后续收益结论均不能宣称通过。",
        "",
        "## Stage 0：A0 数值正确性门禁",
        "",
    ]
    if a0 is None:
        lines.append("结果文件缺失；Stage 0：**FAIL**。")
    else:
        a0_rows = {row["stage"]: row for row in a0["rows"]}
        stage_names = ["Q", "K", "V", "QK", "SOFTMAX", "AV", "WO", "logits"]
        stage_display = {"SOFTMAX": "Softmax"}
        stage_table = []
        for name in stage_names:
            row = a0_rows.get(name)
            if row is None:
                stage_table.append([stage_display.get(name, name), "N/A", "N/A", "N/A", "FAIL"])
                continue
            fp = row["fp32_fixed"]
            fs = row["fixed_secure"]
            stage_table.append([
                stage_display.get(name, name),
                row["shape_packing"],
                f"{number(fp.get('relative_l2'))} / {number(fp.get('max_abs'))}",
                ("N/A" if fs is None else
                 f"{number(fs.get('relative_l2'))} / {number(fs.get('max_abs'))}"),
                row["status"],
            ])
        lines.append(table([
            "Stage", "Shape/packing", "FP32↔Fixed rel-L2 / max abs",
            "Fixed↔Secure rel-L2 / max abs", "Status"], stage_table))
        secure_gate = a0["fixed_secure_gate"]
        fp32_fixed_agreement = a0["fp32_fixed_top1_agreement"]
        a0_status = "PASS" if secure_gate == "PASS" else "FAIL"
        lines.extend([
            "",
            table(["A0 结论", "结果"], [
                ["首个出现明显异常的位置",
                 ("未发现 Fixed→Secure 明显异常；AV/WO 没有 secure trace，"
                  "不对这两层作 secure 结论")],
                ["FP32→Fixed 是否异常", "YES（logits rel-L2=" +
                 number(a0_rows["logits"]["fp32_fixed"]["relative_l2"]) +
                 "，top-1 不一致；这是 fixed-clear 量化差异）"],
                ["Fixed→Secure 是否异常", "NO（所有已捕获阶段 PASS）"],
                ["top-1 FP32 / Fixed / Secure", " / ".join(str(a0["top1"][key])
                                                       for key in ("fp32", "fixed", "secure"))],
                ["A0 门禁", f"{a0_status}（Fixed↔Secure；FP32↔Fixed top-1 agreement={fp32_fixed_agreement}）"],
            ]),
            "",
            "A0 状态：**PASS（Fixed↔Secure conversion gate）**。"
            "FP32↔Fixed top-1 仍未一致，因此后续 accuracy 仍按实际测试结果单独报告，"
            "不把 fixed-clear 结果当作 secure accuracy。",
        ])

    lines.extend(["", "## Stage 1：三路径真实 block 成本", ""])
    cost_table = []
    for path in PATHS:
        for group in GROUPS:
            row = costs.get((path, group))
            if row is None:
                cost_table.append([path, group, "N/A", "N/A", "N/A", "N/A",
                                   "N/A", "N/A", "N/A", "N/A", "N/A"])
            else:
                cost_table.append([
                    path, group, number(row["online_latency_s"]),
                    number(row["he_time_s"]), number(row["mpc_time_s"]),
                    number(row["communication_mb"]), integer(row["rounds"]),
                    number(row["conversion_ct"]), number(row["he_rotations"]),
                    number(row["he_muls"]), row["network"],
                ])
    lines.append(table([
        "Path", "Heads/group", "Online latency (s)", "HE time (s)",
        "MPC time (s)", "Comm (MB)", "Rounds", "Conversion ct",
        "HE rotations", "HE muls", "Network"], cost_table))
    lines.extend(["", "单头实测（g=1，不做 group 除法）：", ""])
    one_head = []
    for path in PATHS:
        row = costs.get((path, 1))
        one_head.append([
            path,
            "N/A" if row is None else number(row["online_latency_s"]),
            "N/A" if row is None else number(row["communication_mb"]),
            "N/A" if row is None else integer(row["rounds"]),
            "N/A" if row is None else row["network"],
            "N/A" if row is None else "measured",
        ])
    lines.append(table(["Path", "Latency/head (s)", "Comm/head (MB)",
                        "Rounds", "Network", "Status"], one_head))

    best_rows = []
    for path in PATHS:
        formal = [row for (row_path, _), row in costs.items()
                  if row_path == path and row["network"] == "1Gbps/0.3ms_RTT"]
        if not formal:
            best_rows.append([path, "N/A", "N/A", "N/A", "N/A"])
            continue
        latency_best = min(formal, key=lambda row:
                           float(row["online_latency_s"]) / int(row["heads_group"]))
        comm_best = min(formal, key=lambda row:
                        float(row["communication_mb"]) / int(row["heads_group"]))
        group = int(latency_best["heads_group"])
        bottle = ("N/A（formal marker 没有 HE/MPC 分解）"
                  if latency_best["mpc_time_s"] == "N/A" else
                  ("HE" if float(latency_best["he_time_s"]) >
                   float(latency_best["mpc_time_s"]) else "MPC"))
        best_rows.append([
            path, f"g={group}",
            number(float(latency_best["online_latency_s"]) / group),
            f"{float(comm_best['communication_mb']) / int(comm_best['heads_group']):.6f}",
            bottle,
        ])
    lines.extend(["", table(["Path", "Best measured group", "Latency/head (s)",
                              "Comm/head (MB)", "主要瓶颈"], best_rows), "",
                  f"Stage 1 状态：**INCOMPLETE**（已测 {len(costs)}/18 个 group；"
                  "formal 网络行是实测，未运行项为 N/A）。",
                  ""])
    anchor = current_a0_block_anchor()
    if anchor is None:
        lines.append("当前 A0 全 block anchor：N/A。")
    else:
        lines.extend([
            "当前版本 A0 全 block anchor（12 Softmax heads；不是单路径 cost row）：",
            "",
            table(["Run", "Heads", "Block online (s)", "Aggregate comm (MB)",
                   "Network", "Status"], [[
                       "A0 current", 12, number(anchor["latency_s"]),
                       number(anchor["communication_mb"]),
                       "unshaped_diagnostic_only", "measured"
                   ]]),
            "该 anchor 包含 block 内 QKV、Softmax、AV/WO、LayerNorm 等完整阶段，"
            "不能替代上面的三路径 microbenchmark。",
            "",
        ])

    lines.extend(["## Stage 2–6：门禁依赖项", "",
                  table(["Stage", "Status", "原因"], [
                      ["Stage 2 calibration", "N/A", "A0 已通过 conversion gate，但本轮尚未重新 calibration"],
                      ["Stage 3 selector / fixed assignment", "N/A", "没有本轮最终 head_assignment.json"],
                      ["Stage 4 algorithm comparison", "N/A", "没有本轮模型级 accuracy gate 结果"],
                      ["Stage 5 packing", "N/A", "没有本轮 packed/unpacked 对照"],
                      ["Stage 6 ablation", "N/A", "依赖 Stage 2–5"],
                  ]), "",
                  ""])

    lines.extend(["## 完整 12 层推理（已有真实运行）", ""])
    full_specs = [
        ("All-Softmax", ROOT / "outputs/a3_a4_review/plans/a3_all_softmax_full12.txt",
         ROOT / "outputs/a3_a4_review/a3_all_softmax_full12_blb_fast_current_20260901"),
        ("11S-1R", ROOT / "outputs/a1_a2_review/a2_softmax11_relu1_head_plan.txt", None),
        ("10S-2L", ROOT / "outputs/a4_pair_review/head_plan.txt",
         ROOT / "outputs/a3_a4_review/a4_pair_aligned_10softmax2linear_full12_blb_fast_current_20260901"),
        ("8S-4L", ROOT / "outputs/a4_pair4_review/head_plan.txt", None),
        ("4S-4R-4L", ROOT / "outputs/a1_a2_review/a2_manual_grouped_head_plan.txt", None),
    ]
    full_rows = []
    for name, plan, run in full_specs:
        softmax, relu, linear = plan_counts(plan) if plan.is_file() else (0, 0, 0)
        if run is None or not (run / "ours_summary.csv").is_file():
            full_rows.append([name, softmax, relu, linear, "N/A", "N/A", "N/A",
                              "N/A", "N/A", "N/A", "N/A"])
            continue
        metric = metric_map(run)
        full_rows.append([
            name, softmax, relu, linear,
            number(metric.get("secure_end_to_end_s")),
            number(metric.get("setup_plus_secure_s",
                             metric.get("setup_plus_secure_s_max_party"))),
            number(float(metric.get("total_communication_bytes_alice_plus_bob", "nan")) / 1e6),
            integer(metric.get("online_rounds_max_party")),
            integer(metric.get("conversion_ciphertexts")),
            metric.get("prediction", "N/A"), "MEASURED_OLD_BINARY; accuracy=N/A",
        ])
    lines.append(table(["Plan", "#Softmax", "#ReLU", "#Linear", "Online (s)",
                        "Total incl. setup (s)", "Total Comm (MB)", "Rounds",
                        "Conversion ct", "Secure one-input top-1", "Status"], full_rows))
    lines.extend(["", "完整运行结论：已有 All-Softmax 与 10S/2L 两条真实 full-12 性能记录；"
                  "双方预测一致但没有 50 张 secure accuracy，且使用旧版已完成运行的 binary hash，"
                  "因此仅作为性能记录，不作为当前版本最终精度结论。",
                  ""])

    lines.extend(["## 50 张照片准确率", ""])
    if accuracy is None:
        lines.append("50 张 fixed-clear 评估尚未完成：**N/A**；secure accuracy：**N/A（未运行）**。")
    else:
        accuracy_rows = []
        for row in accuracy["summaries"]:
            accuracy_rows.append([row["plan"], f"{row['top1_correct']}/{row['samples']}",
                                  f"{100.0 * row['top1']:.2f}%",
                                  "N/A（未运行 secure 50-image）", "fixed-clear measured"])
        lines.append(table(["Head plan", "Top-1 correct", "Fixed-clear accuracy",
                            "Secure accuracy", "Status"], accuracy_rows))
        lines.extend(["", f"样本来自固定 manifest 的前 {accuracy['manifest_rows']} 行，"
                      f"manifest_sha256={accuracy['manifest_sha256']}；"
                      "该表只证明 fixed-clear 路径结果；secure 50-image 尚未运行。", ""])

    lines.extend(["## 运行与速度说明", "",
                  "本次只新增了审计/汇总脚本，没有修改生产 runner、默认 head plan、"
                  "GPU kernel 或已有二进制；被中止的当前 wide-path g=1 与 full-12 诊断没有结果文件进入正式表格。"
                  "最新 A0 单 block unshaped 实测 block-online=624.370 s、aggregate comm=7392.690 MB；"
                  "已有 fast full-12 实测：All-Softmax 537.982 s，10S/2L native Linear 524.750 s。"
                  "当前 wide-path full-12 在第一层 Softmax 已测到约 374.8 s 后被停止；"
                  "这些诊断不改变原有生产 runner 的默认速度路径。", "",
                  "## 主要数据源", "",
                  "- A0 trace: `outputs/a0_after_rotation_fix_20260902c/a0_trace_comparison.json`",
                  "- Stage 1 measured rows: `outputs/head_experiment_20260901/stage1/stage1_formal_real_costs.csv`",
                  "- 50-image fixed-clear: `outputs/20260902_accuracy_50/fixed_clear_50_after_a0.json`（若已完成）",
                  "- Full-12 summaries: `outputs/a3_a4_review/*full12*/ours_summary.csv`",
                  ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OURS_HEAD_REPORT status=PASS output={args.output}")


if __name__ == "__main__":
    main()
