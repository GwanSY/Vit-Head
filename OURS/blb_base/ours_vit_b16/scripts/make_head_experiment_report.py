#!/usr/bin/env python3
"""Write the staged ViT head-experiment progress report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


GROUP_SIZES = (1, 2, 4, 6, 8, 12)
PATHS = ("Softmax", "ReLU", "Linear")
CALIBRATION_PROTOCOLS = {
    "relu_norm_mpc": "ReLU",
    "he_linear": "Linear",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str] | None, name: str) -> str:
    if row is None or not row.get(name) or row[name] == "N/A":
        return "N/A"
    if name.endswith("_s") or name == "communication_mb":
        return f"{float(row[name]):.6f}"
    return row[name]


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def stage1_table(rows: list[dict[str, str]]) -> str:
    lookup = {(row["path"], int(row["heads_group"])): row for row in rows
              if row["network"] == "1Gbps/0.3ms_RTT"}
    output = []
    for path in PATHS:
        for group in GROUP_SIZES:
            row = lookup.get((path, group))
            output.append([
                path, group,
                value(row, "online_latency_s"), value(row, "he_time_s"),
                value(row, "mpc_time_s"), value(row, "communication_mb"),
                value(row, "rounds"), value(row, "conversion_ct"),
                value(row, "he_rotations"), value(row, "he_muls"),
            ])
    return md_table(
        ["Path", "Heads/group", "Online latency (s)", "HE time (s)",
         "MPC time (s)", "Comm (MB)", "Rounds", "Conversion ct",
         "HE rotations", "HE muls"], output)


def stage1_summary(rows: list[dict[str, str]]) -> str:
    lookup = defaultdict(list)
    for row in rows:
        if row["network"] == "1Gbps/0.3ms_RTT":
            lookup[row["path"]].append(row)
    output = []
    bottlenecks = {
        "Softmax": "MPC nonlinear + conversion/round trips",
        "ReLU": "MPC reciprocal/normalization",
        "Linear": "native CKKS QK/AV/WO",
    }
    for path in PATHS:
        measured = lookup[path]
        if not measured:
            output.append([path, "N/A", "N/A", "N/A", "N/A"])
            continue
        best = min(measured, key=lambda row:
                   float(row["online_latency_s"]) / int(row["heads_group"]))
        output.append([
            path, best["heads_group"],
            f"{float(best['online_latency_s']) / int(best['heads_group']):.6f}",
            f"{float(best['communication_mb']) / int(best['heads_group']):.6f}",
            bottlenecks[path],
        ])
    return md_table(["Path", "最佳 group size", "latency/head", "Comm/head",
                     "主要瓶颈"], output)


def calibration_tables(path: Path, threshold: float) -> tuple[str, str]:
    records = defaultdict(dict)
    for row in read_csv(path):
        key = (int(row["layer"]), int(row["head"]), row["protocol"])
        records[key][row["oracle"]] = float(row["relative_l2"])
        if row["protocol"] == "he_linear":
            records[key]["alpha"] = float(row["alpha"])

    detail = []
    summary = []
    for layer in range(12):
        sensitive, relu_candidates, linear_candidates = [], [], []
        for head in range(12):
            values = {}
            for protocol, label in CALIBRATION_PROTOCOLS.items():
                values[label] = max(records[(layer, head, protocol)].get("fp32", 0.0),
                                     records[(layer, head, protocol)].get("fixed", 0.0))
            relu_ok = values["ReLU"] <= threshold
            linear_ok = values["Linear"] <= threshold
            if not relu_ok and not linear_ok:
                sensitive.append(head)
            if relu_ok:
                relu_candidates.append(head)
            if linear_ok:
                linear_candidates.append(head)
            alpha = records[(layer, head, "he_linear")].get("alpha")
            detail.append([
                layer, head, f"{values['ReLU']:.6f}",
                f"{values['Linear']:.6f}",
                f"{alpha:.8g}" if alpha is not None else "N/A",
                "YES*" if relu_ok else "NO*",
                "YES*" if linear_ok else "NO*",
            ])
        fmt = lambda values: ",".join(str(v) for v in values) if values else "—"
        summary.append([layer, fmt(sensitive), fmt(relu_candidates),
                        fmt(linear_candidates)])
    return (
        md_table(["Layer", "Head", "ReLU rel-L2", "Linear rel-L2", "alpha_h",
                  "ReLU可接受", "Linear可接受"], detail),
        md_table(["Layer", "Softmax-sensitive Heads", "ReLU候选 Heads",
                  "Linear候选 Heads"], summary),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-costs", type=Path, required=True)
    parser.add_argument("--smoke-costs", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--selector-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = read_csv(args.formal_costs)
    smoke = read_csv(args.smoke_costs)
    detail, summary = calibration_tables(args.calibration, 0.25)
    trace = json.loads(args.selector_trace.read_text(encoding="utf-8"))

    stage0_rows = [[name, "N/A", "N/A", "N/A", "N/A", "N/A"]
                   for name in ("Q", "K", "V", "QK", "Softmax", "AV", "WO", "logits")]
    stage0 = md_table(["Stage", "Shape/packing", "FP32↔Fixed rel-L2",
                       "Fixed↔Secure rel-L2", "max abs error", "Status"],
                      stage0_rows)
    smoke_rows = [[row["path"], row["heads_group"], row["online_latency_s"],
                   row["he_time_s"], row["mpc_time_s"], row["communication_mb"],
                   row["rounds"], row["conversion_ct"], row["he_rotations"],
                   row["he_muls"]] for row in smoke]
    smoke_table = md_table(
        ["Path", "Heads/group", "Online latency (s)", "HE time (s)",
         "MPC time (s)", "Comm (MB)", "Rounds", "Conversion ct",
         "HE rotations", "HE muls"], smoke_rows or [["N/A"] * 10])

    na_methods = [[method, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                   "N/A", "N/A"] for method in
                  ("All-Softmax", "Random", "Sensitivity-only", "Cost-only", "Ours")]
    method_table = md_table(
        ["Method", "#Softmax", "#ReLU", "#Linear", "Accuracy",
         "Accuracy drop", "Predicted latency", "Measured latency", "Comm"],
        na_methods)
    ablation_rows = [[name, auto, conv, pack, linear, "N/A", "N/A", "N/A", "N/A", "N/A"]
                     for name, auto, conv, pack, linear in (
                         ("A0 All-Softmax", "NO", "NO", "NO", "NO"),
                         ("A2 Selected", "YES", "NO", "NO", "NO"),
                         ("A3 Selected+Conversion", "YES", "YES", "NO", "NO"),
                         ("A3+Packing", "YES", "YES", "YES", "NO"),
                         ("A4 Full", "YES", "YES", "YES", "YES"))]
    ablation_table = md_table(
        ["Variant", "自动选择", "Selective conversion", "Group packing",
         "CKKS Linear", "Accuracy", "Online time", "Total time", "Comm",
         "Conversion ct"], ablation_rows)
    stage7_rows = [[name] + ["N/A"] * 9 for name in
                   ("All-Softmax baseline", "Ours Accuracy-first",
                    "Ours Balanced", "Ours Performance-first")]
    stage7_table = md_table(
        ["Method", "Accuracy", "Top-1 agreement", "Total latency",
         "Online latency", "HE time", "MPC time", "Total Comm", "Rounds",
         "Conversion ct"], stage7_rows)
    packing_rows = [["Unpacked", "N/A", "NO"] + ["N/A"] * 8,
                    ["Packed", "N/A", "YES"] + ["N/A"] * 8]
    packing_table = md_table(
        ["Variant", "Assignment相同", "Group packing", "Online latency",
         "HE time", "MPC time", "Comm", "Rounds", "Conversion ct",
         "MPC calls", "HE rotations"], packing_rows)
    packing_correctness = md_table(
        ["Tensor", "Packed↔Unpacked rel-L2", "max abs", "Status"],
        [[name, "N/A", "N/A", "N/A"] for name in ("QK", "Norm output", "AV", "WO", "logits")])

    lines = [
        "# ViT-B/16 头级异构安全注意力实验进度",
        "",
        "本报告严格按阶段记录真实运行结果。当前 A0 仍为 **FAIL**；因此后续结果只能作为诊断数据，不能生成最终 Head assignment，也不能据此运行完整 12 层。",
        "",
        "## 速度保护",
        "",
        "默认转换入口固定使用原有 BLB 本地 FFT；实验性 secure/field-FFT conversion 已删除。当前单层 smoke 日志出现 `BLB_FAST_LOCAL_FFT`，未运行完整 12 层，不改变既定 vit-gpu 约 8 分钟、vit-cpu 约 12 分钟配置。",
        "",
        "## Stage 0：A0 数值正确性门禁",
        "",
        "冻结一致性修复后没有重跑 A0 tensor trace，按要求未运行项记 N/A；A0 仍由此前真实 secure conversion round-trip 失败判定。",
        "",
        stage0,
        "",
        md_table(["A0结论", "结果"], [
            ["首个出现明显异常的位置", "CKKS↔MPC secure conversion boundary"],
            ["FP32→Fixed 是否异常", "是（历史 fixed-clear trace，冻结后未复测）"],
            ["Fixed→Secure 是否异常", "是（真实 secure round-trip）"],
            ["top-1 FP32 / Fixed / Secure", "733 / 899 / 947（历史诊断值，不作为当前 accuracy）"],
            ["A0 门禁", "FAIL"],
        ]),
        "",
        "## Stage 1：三路径真实成本测量",
        "",
        "正式表只收录 1 Gbps/0.3 ms、两张物理 GPU、成功退出的真实运行。没有用成本外推填充缺失 group。当前环境无法执行 `tc netem`（RTNETLINK Operation not permitted），所以新 smoke 单独列出，不进入正式 Cost Model。",
        "",
        stage1_table(formal),
        "",
        "### 当前已测路径的 group 摘要",
        "",
        stage1_summary(formal),
        "",
        "### 新编译快速路径 unshaped smoke（不计入正式表）",
        "",
        smoke_table,
        "",
        "Stage 1 状态：**FAIL/BLOCKED（A0 FAIL，且 formal group grid 不完整）**；已收集的 9 行仅作真实诊断数据。",
        "",
        "## Stage 2：Head 敏感度 calibration",
        "",
        "使用 4 个固定公开 calibration input，以 Softmax head context 为 reference；Linear alpha_h 为离线 least-squares 拟合。`*` 只表示局部 relative-L2 ≤ 0.25 候选，不是模型级接受。",
        "",
        detail,
        "",
        summary,
        "",
        "Stage 2 状态：**PASS（公开数据局部 calibration）**；最终替换资格：**BLOCKED（A0 FAIL，尚未执行模型级 logits/accuracy gate）**。",
        "",
        "## Stage 3：安全成本感知 Head 路径选择",
        "",
        "已实现无枚举 greedy selector：从 All-Softmax 出发，用实测 group cost 与局部 disturbance 计算 score，并要求每一步提供 model-level metric gate。由于 A0 FAIL，本轮只输出 blocked trace，不生成 `head_assignment.json`。",
        "",
        md_table(["Layer", "Softmax Heads", "ReLU Heads", "Linear Heads", "Predicted cost", "Calibration accuracy/drop"],
                 [["0–11", "N/A", "N/A", "N/A", "N/A", "N/A"]]),
        "",
        md_table(["Step", "Layer.Head", "From→To", "Disturbance", "Cost saved", "Score", "Model metric", "Accepted"],
                 [["—", "N/A", "N/A", "N/A", "N/A", "N/A", "BLOCKED_A0", "NO"]]),
        "",
        f"selector trace：`{args.selector_trace}`；status=`{trace.get('status', 'N/A')}`。",
        "",
        "Stage 3 状态：**FAIL/BLOCKED（A0 FAIL）**。",
        "",
        "## Stage 4：路径选择算法验证",
        "",
        "没有最终固定 assignment，也没有相同 accuracy budget 的模型级门禁结果；因此不使用已有任意 assignment 的性能值代替比较。",
        "",
        method_table,
        "",
        md_table(["Cost model验证", "数值"], [
            ["latency 平均预测误差", "N/A"],
            ["communication 平均预测误差", "N/A"],
            ["预测排序与实测排序是否一致", "N/A"],
        ]),
        "",
        "Stage 4 状态：**N/A（A0 FAIL）**。",
        "",
        "## Stage 5：同路径 Multi-Head Grouping / Packing",
        "",
        "当前可复核的路径结果本身已经是 grouped/packed kernel；没有同一 assignment 的 unpacked 实现和真实对照运行，因此不把 packed 单点结果写成 packing 收益。",
        "",
        packing_table,
        "",
        md_table(["Packing收益", "数值"], [[name, "N/A"] for name in (
            "Latency reduction", "Communication reduction", "Conversion reduction",
            "MPC-call reduction", "HE rotation reduction")]),
        "",
        packing_correctness,
        "",
        "Stage 5 状态：**N/A（没有 unpacked 对照，且 A0 FAIL）**。",
        "",
        "## Stage 6：最终消融",
        "",
        ablation_table,
        "",
        "Stage 6 状态：**N/A（A0 FAIL）**。",
        "",
        "## Stage 7：完整 12 层 ViT",
        "",
        "本轮没有运行完整 12 层。",
        "",
        stage7_table,
        "",
        "Stage 7 状态：**N/A（A0、三路径数值、packing gate 未全部 PASS）**。",
        "",
        "## 可复核文件",
        "",
        f"- formal Stage 1 rows：`{args.formal_costs}`",
        f"- unshaped smoke rows：`{args.smoke_costs}`（不计入正式模型）",
        f"- Stage 2 calibration：`{args.calibration}`",
        f"- Stage 3 selector trace：`{args.selector_trace}`",
        "- 计划生成：`ours_vit_b16/scripts/make_three_path_profiler_plans.py`",
        "- profiler 收集：`ours_vit_b16/scripts/collect_three_path_costs.py`",
        "- selector：`ours_vit_b16/scripts/greedy_head_selector.py`",
        "",
        "最终结论：目前只能确认局部公开 calibration 和部分真实安全 path 成本；不能据此声称成本感知 assignment、packing 或完整方案优于 Random/Sensitivity-only，也不能报告完整 12 层改善。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OURS_HEAD_EXPERIMENT_REPORT status=PASS output={args.output}")


if __name__ == "__main__":
    main()
