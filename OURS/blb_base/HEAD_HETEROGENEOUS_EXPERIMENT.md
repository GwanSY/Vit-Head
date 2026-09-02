# ViT-B/16 头级异构安全注意力实验

> 当前续跑结果以 [2026-09-01 progress report](outputs/head_experiment_20260901/HEAD_HETEROGENEOUS_EXPERIMENT_PROGRESS.md) 为准。已完成部分公开 calibration、部分正式真实 path 成本和 selector 阻断 trace；当前 A0 仍 FAIL，Stage 3–7 不得生成最终 assignment 或运行完整 12 层。

## 当前结论

当前主线仍被 Stage 0 的 A0 门禁 **FAIL** 阻断。本轮没有运行完整 12 层或生成最终 Head assignment；后续续跑已完成部分真实 path profiler、公开 calibration 和 selector 阻断 trace，详见上方 progress report。

为保护已经确定的推理速度基线（vit-gpu 约 8 分钟、vit-cpu 约 12 分钟），已从生产路径移除 field-FFT、inverse-root cache、61-bit FFT scale 和宽位宽 conversion 分支，恢复原始 BLB 本地 FFT conversion。只保留了不改变协议计算量的 worker 端口步长修复。编译结果为 PASS，当前二进制中没有 field-FFT 字符串或 61-bit FFT scale。

## Stage 0：A0 数值正确性门禁

### 0A. 当前有效门禁表

下面的模型级 secure tensor 表没有在恢复快速生产路径后再次运行；因此按要求记为 N/A。A0 已经由独立的真实 CKKS→MPC→CKKS secure round-trip 失败而判定 FAIL。

| Stage | Shape/packing | FP32↔Fixed rel-L2 | Fixed↔Secure rel-L2 | max abs error | Status |
|---|---|---:|---:|---:|---|
| Q | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |
| K | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |
| V | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |
| QK | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |
| Softmax | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |
| AV | N/A（当前 trace 未输出） | N/A | N/A | N/A | N/A |
| WO | N/A（当前 trace 未输出） | N/A | N/A | N/A | N/A |
| logits | N/A（恢复路径后未重跑） | N/A | N/A | N/A | N/A |

### 0B. 已完成的真实 conversion 边界诊断

诊断使用 1 个真实 CKKS ciphertext、双方真实 secure conversion 和公开确定性 fixture；它不是模型精度结果，也没有把输出重新送回 secure graph。

| Candidate | FFT scale / working width | max abs error (raw) | bad values | Result |
|---|---:|---:|---:|---|
| 原始 BLB 快速路径的宽位宽候选 | 40 / 77 | 224 | 14,223 / 16,384 | FAIL |
| 宽位宽候选（scale 48） | 48 / 77 | 227 | 14,222 / 16,384 | FAIL |
| 宽位宽候选（scale 61） | 61 / 65 | 228 | 14,204 / 16,384 | FAIL |
| field-FFT 候选（无中心偏移） | 40 / 77 | 1,066,233,734 | 16,384 / 16,384 | FAIL；已删除 |

以上记录来自 `outputs/a0_conversion_selftest_fixed_20260901`。宽位宽候选的单 ciphertext 约需 2–3 分钟，field-FFT 候选同样达到分钟级，因此均不进入生产路径。

### 0C. 早期 fast-path 模型 trace（仅定位，不作为当前 gate）

此前的 1-block fast-path trace 已显示异常从 conversion 后的早期 attention tensor 放大；该 trace 在本轮清理前生成，不能替代恢复路径后的正式门禁。

| Tensor | Shape | Fixed↔Secure rel-L2（pre-fix diagnostic） | max abs |
|---|---|---:|---:|
| Q | 256×768 | ≈5.01759e7 | ≈1.2361e7 |
| K | 256×768 | ≈6.86777e6 | N/A |
| V | 256×768 | ≈7.28761e7 | N/A |
| QK | 12×256×256 | ≈2.23507e6 | N/A |
| Softmax | 12×256×256 | ≈3.02486e8 | N/A |
| AV | N/A | N/A | N/A |
| WO | N/A | N/A | N/A |
| logits | N/A | N/A | N/A |

### A0 结论

| A0结论 | 结果 |
|---|---|
| 首个出现明显异常的位置 | CKKS↔MPC conversion 的本地 fixed-point FFT / additive-share round-trip 边界 |
| FP32→Fixed 是否异常 | 是；历史 fixed-clear 诊断 top-1 为 FP32 733、Fixed 899；该数值也需要在 conversion 修复后复测 |
| Fixed→Secure 是否异常 | 是；真实 secure round-trip 的 max abs error 为 224–228 raw |
| top-1 FP32 / Fixed / Secure | 733 / 899 / 947（历史诊断值，非本轮有效结果） |
| A0 门禁 | **FAIL** |

A0 FAIL，因此本报告不使用已有 calibration 生成最终 assignment。

## Stage 1：三路径真实成本测量

入口条件 A0 PASS 未满足，未运行任何三路径 microbenchmark；没有成本外推。

| Path | Heads/group | Online latency (s) | HE time (s) | MPC time (s) | Comm (MB) | Rounds | Conversion ct | HE rotations | HE muls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Softmax | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Softmax | 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Softmax | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Softmax | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Softmax | 12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ReLU | 12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Linear | 12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

| Path | 最佳 group size | latency/head | Comm/head | 主要瓶颈 |
|---|---:|---:|---:|---|
| Softmax | N/A | N/A | N/A | N/A |
| ReLU | N/A | N/A | N/A | N/A |
| Linear | N/A | N/A | N/A | N/A |

**Stage 1 状态：N/A（A0 FAIL 阻断，未形成 Cost Model）。**

## Stage 2：Head 敏感度 calibration

固定 calibration set 未在 A0 通过前使用。下表用每层聚合行表示所有 12 个 head 均未运行。

| Layer | Head | ReLU rel-L2 | Linear rel-L2 | alpha_h | ReLU可接受 | Linear可接受 |
|---:|---|---:|---:|---:|---|---|
| 0 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 1 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 2 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 3 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 4 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 5 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 6 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 7 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 8 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 9 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 10 | 0–11 | N/A | N/A | N/A | N/A | N/A |
| 11 | 0–11 | N/A | N/A | N/A | N/A | N/A |

| Layer | Softmax-sensitive Heads | ReLU候选 Heads | Linear候选 Heads |
|---:|---|---|---|
| 0–11 | N/A | N/A | N/A |

**Stage 2 状态：N/A（A0 FAIL 阻断）。**

## Stage 3：安全成本感知 Head 路径选择

没有读取旧 calibration 生成 assignment，没有枚举 `3^H`，也没有动态选择。

| Layer | Softmax Heads | ReLU Heads | Linear Heads | Predicted cost | Calibration accuracy/drop |
|---:|---:|---:|---:|---:|---:|
| 0–11 | N/A | N/A | N/A | N/A | N/A |

| Step | Layer.Head | From→To | Disturbance | Cost saved | Score | Model metric | Accepted |
|---:|---|---|---:|---:|---:|---:|---|
| — | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

`head_assignment.json`：本轮未生成。**Stage 3 状态：N/A（A0 FAIL 阻断）。**

## Stage 4：路径选择算法验证

| Method | #Softmax | #ReLU | #Linear | Accuracy | Accuracy drop | Predicted latency | Measured latency | Comm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All-Softmax | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Random | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Sensitivity-only | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Cost-only | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ours | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

| Cost model验证 | 数值 |
|---|---:|
| latency 平均预测误差 | N/A |
| communication 平均预测误差 | N/A |
| 预测排序与实测排序是否一致 | N/A |

**Stage 4 状态：N/A（A0 FAIL 阻断）。**

## Stage 5：同路径 Multi-Head Grouping / Packing

没有可冻结的 Stage 3 assignment，因此未运行 unpacked/packed 对照。

| Variant | Assignment相同 | Group packing | Online latency | HE time | MPC time | Comm | Rounds | Conversion ct | MPC calls | HE rotations |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpacked | N/A | NO | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Packed | N/A | YES | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

| Packing收益 | 数值 |
|---|---:|
| Latency reduction | N/A |
| Communication reduction | N/A |
| Conversion reduction | N/A |
| MPC-call reduction | N/A |
| HE rotation reduction | N/A |

| Tensor | Packed↔Unpacked rel-L2 | max abs | Status |
|---|---:|---:|---|
| QK | N/A | N/A | N/A |
| Norm output | N/A | N/A | N/A |
| AV | N/A | N/A | N/A |
| WO | N/A | N/A | N/A |
| logits | N/A | N/A | N/A |

**Stage 5 状态：N/A（A0 FAIL 阻断）。**

## Stage 6：最终消融

| Variant | 自动选择 | Selective conversion | Group packing | CKKS Linear | Accuracy | Online time | Total time | Comm | Conversion ct |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| A0 All-Softmax | NO | NO | NO | NO | N/A | N/A | N/A | N/A | N/A |
| A2 Selected | YES | NO | NO | NO | N/A | N/A | N/A | N/A | N/A |
| A3 Selected+Conversion | YES | YES | NO | NO | N/A | N/A | N/A | N/A | N/A |
| A3+Packing | YES | YES | YES | NO | N/A | N/A | N/A | N/A | N/A |
| A4 Full | YES | YES | YES | YES | N/A | N/A | N/A | N/A | N/A |

**Stage 6 状态：N/A（A0 FAIL 阻断）。**

## Stage 7：完整 12 层 ViT

完整 12 层运行条件未满足，本轮明确未运行。

| Method | Accuracy | Top-1 agreement | Total latency | Online latency | HE time | MPC time | Total Comm | Rounds | Conversion ct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All-Softmax baseline | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ours Accuracy-first | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ours Balanced | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ours Performance-first | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Stage 7 状态：N/A（A0 FAIL，未达到运行条件）。**

## 可复核文件与速度保护

- 编译：`scripts/build_vit_b16.sh`，结果 `OURS_VIT_BUILD status=PASS`。
- 当前生产转换：`ours_vit_b16/runtime/ckks_bert_basic.cpp` 的原始 BLB `ckks_to_mpc` / `mpc_to_ckks`。
- 当前 FFT scale：`ours_vit_b16/runtime/linear_ckks.cpp` 中为 40。
- A0 真实 conversion 诊断：`outputs/a0_conversion_selftest_fixed_20260901/`。
- 早期 1-block trace：`outputs/a0_stage0_current_20260901_rerun/`。
- 旧的 12 层 A3/A4 结果没有作为本轮结果复用；它们是在 A0 未通过时产生的性能诊断，不能支撑最终 accuracy 或 assignment 结论。

### 最终判断

在不保留分钟级慢 conversion 路径、也不影响既有 vit-gpu/vit-cpu 速度基线的前提下，本轮无法完成 A0 修复。因此如实停止在 Stage 0；后续成本曲线、selector、packing 和完整 12 层结果均不报告伪造或外推数值。
