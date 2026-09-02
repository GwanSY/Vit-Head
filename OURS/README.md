# ViT head-level heterogeneous secure inference

当前生产实现位于 `blb_base/`。它是从
`/home/dm/mind/Vit-Head/BLB` 的优化后 ViT-B/16 Phantom GPU CKKS
baseline 直接建立的 clean fork，不复用旧 OURS 的 65-token CKKS 原型、
BFV/BOLT adapter、head assignment 或异构内核。

## 当前布局

- `blb_base/gpu/SCI`：BLB 使用的 SCI、Phantom GPU 和 CKKS/MPC 后端。
- `blb_base/vit_b16`：不可变 A0 donor runtime；用于持续对照。
- `blb_base/ours_vit_b16`：从 A0 直接分出的异构实现，固定使用
  197 logical / 256 physical tokens、12 层、12 heads × 64。
- `blb_base/scripts`：构建、两方运行、汇总和验证脚本。
- `blb_base/checkpoints/vit_b16`、`blb_base/inputs/vit_b16`：指向 BLB
  原始只读模型和输入的相对链接，避免复制 1.1 GB checkpoint。

旧 65/128-token runtime、BFV adapter、旧 head assignment、历史 build/output
均不属于当前实现，也不参与配置或链接。

## 构建 A0

```bash
cd /home/dm/mind/Vit-Head/OURS/blb_base
bash scripts/build_vit_b16.sh
```

可用 `CMAKE_BIN=/path/to/cmake` 覆盖 CMake。脚本要求 CMake >= 3.17，
并使用仓库内 vendored Eigen，避免 donor 容器的失效绝对路径。

构建目标：

```text
blb_base/build/gpu/bin/ckks_vit_b16_main       # 不可变 A0
blb_base/build/gpu/bin/ours_vit_b16_main      # 新异构开发线
blb_base/build/gpu/bin/ours_vit_hetero_plan_test
```

## A0–A4

- A0：原始 BLB ViT-B/16，所有 heads 使用 secure Softmax。
- A1：固定的无训练 head assignment；保持保守的全量跨域执行。
- A2：增加同步 head reorder 和 protocol-aware grouping。
- A3：Linear heads 保持 CKKS，真实跳过 CKKS-to-MPC-to-CKKS。
- A4：增加 alpha 融合、HE-linear 批处理和实测 cost gating。

各版本必须使用相同模型、输入、GPU/网络设置和统计口径。A0 数值正确性
未通过前，不允许把异构版本标记为正确或报告最终 speedup。

## 当前可用接口

`ours_vit_b16_main` 接受 `--ablation a0..a4` 与 `--head-plan FILE`。
A0 使用内置全 Softmax 原顺序计划；A1–A4 必须提供公开离线计划。当前 A1
执行逐 head、全量跨域归一化；A2 会同步重排 WQ/WK/WV/偏置和 WO，并把同协议
连续 heads 合并调用。A2 启动密态两方前会对 exact model/input/plan 强制运行
FP32 + fixed reorder oracle。A3/A4 现在执行真实的当前 BLB-fast 路径；若计划
无效会 fail-closed，不会回退到 A2/A0。A0 数值门禁仍关闭时，A3/A4 只能作为
结构/性能诊断，不能宣称准确率。

公开数据校准入口：

```bash
python3 -B blb_base/ours_vit_b16/scripts/calibrate_heterogeneous_heads.py \
  --model blb_base/checkpoints/vit_b16/bolt_model.txt \
  --manifest calibration.csv \
  --output-dir blb_base/outputs/calibration
```

没有真实 Phantom GPU 三路径 cost CSV 时仅输出候选误差；要生成最终 plan，
至少需要 4 个公开样本、明确的 logits/accuracy 阈值和实测 cost 文件。格式见
`blb_base/HEAD_PLAN_FORMAT.md`。
