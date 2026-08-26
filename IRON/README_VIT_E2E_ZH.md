# IRON 端到端 ViT 密态推理

本实现补齐了 IRON 仓库原先缺失的密态 ViT 执行图，可以从 BOB 的私有图像和
ALICE 的私有模型出发，依次执行 Patch Embedding、完整 Transformer block、最终
LayerNorm、分类器与安全 Argmax，最后只向 BOB 输出预测类别。

实现只复用仓库已经存在的 IRON BERT/SCI 算子和 SEAL BFV 线性计算方法，不新增
密码协议，也不提出新的近似算法。IRON 原仓库虽然包含 CCT/ViT 的 PyTorch、
TensorFlow 和定点明文 C++，但其 README 明确说明只发布了明文实验脚本，没有可运行
的密态 ViT。因此这里用已有 BERT 密态算子补齐标准 pre-LN ViT 图。

> `IRON_VIT_GENERIC` 继续保留固定随机种子的 tiny fixture，用于兼容原有 IRON
> smoke 回归；它不是精度模型。标准交付路径是下面的真实 ViT-B/16 checkpoint，
> 使用 `IRON_VIT` 的 197→256 compact packing 图。

## 当前标准 ViT-B/16 路径

已接入真实 `google/vit-base-patch16-224` checkpoint（SHA-256
`1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d`）：

```text
224x224x3 -> 196 patches + CLS = 197 logical tokens
197 logical tokens -> 256 physical tokens (positions 197..255 are padding)
hidden=768, heads=12, head_dim=64, FFN=3072, blocks=12, classes=1000
pre-LN: LN -> QKV -> QK^T -> public 197x197 mask -> MPC Softmax
      -> Softmax*V -> output projection -> residual
      -> LN -> GELU -> FFN -> residual -> final LN -> CLS classifier
```

patchify 在 BOB 本地完成；模型、CLS、position、bias、所有 block 权重和分类头只
由 ALICE 加载。padding mask 在 Softmax 前按公开 197×197 有效区域应用，并在
embedding、每个 block 边界和输出裁剪处应用公开 token mask。生产 binary 只输出
最终类别，绝不恢复中间明文；`IRON_VIT_VALIDATE` 只供最终 logits/aggregate-only
验收使用。

可复现构建命令（工作区根目录执行）：

```bash
cd /home/dm/mind/MIND_code/Pencil
JOBS=4 bash IRON/scripts/build_secure_vit.sh
```

完整真实图片 secure 推理命令：

```bash
cd /home/dm/mind/MIND_code/Pencil
IMAGE="$PWD/BOLT/outputs/vit_b16_smoke/parrots.png" \
MODEL="$PWD/BOLT/checkpoints/vit_b16/bolt_model.txt" \
MANIFEST="$PWD/BOLT/checkpoints/vit_b16/manifest.json" \
OUTPUT="$PWD/IRON/outputs/vit_b16_e2e" PORT=49010 \
  bash IRON/scripts/run_vit_b16_e2e.sh
```

loader 同时接受 `BOLT_VIT_MODEL_V{1,2,3}` 和 `IRON_VIT_MODEL_V{1,2,3}`，并支持
合并/独立 Q、K、V 的 checkpoint converter 输入。真实 B/16 的 shape、scale、bit
length、packing、HE/MPC/conversion 计时在输出目录的 `run_metadata.json`、
`operator_trace.csv`、`performance.csv`、`backend_breakdown.csv`；验收 observer
只输出 `secure_aggregate_errors.csv` 中的 5（单 block）或 27（12 block）个误差
聚合行。

tiny 兼容 smoke 仍使用：

```bash
PORT=47100 OUTPUT="$PWD/IRON/outputs/vit_smoke" \
  bash IRON/scripts/run_secure_vit_e2e.sh
```

正式 ImageNet validation 使用本地 `BOLT/datasets/imagenet2012/val` 的 50,000 张图片。
可复现的浮点 checkpoint 评估命令为：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  BOLT/.venv-vit-b16/bin/python \
  BLB/vit_b16/scripts/evaluate_vit_b16_pytorch.py \
  --checkpoint-dir BOLT/checkpoints/vit_b16_hf \
  --manifest BOLT/datasets/imagenet2012/val/validation_manifest.csv \
  --image-root BOLT/datasets/imagenet2012/val \
  --output IRON/outputs/vit_b16_imagenet_pytorch_50k.json \
  --batch-size 64 --progress-every 5000
```

当前实测 PyTorch/reference top-1 为 `0.803260`（40163/50000），top-5 为
`0.954860`（47743/50000）。这是浮点/reference accuracy，不标作 secure accuracy；
secure 50k 多图片评估尚未运行。

## 1. 加密方法

这里沿用 IRON BERT 的半诚实两方 **HE + MPC** 分工：

- ALICE（`r=1`）持有私有模型参数；BOB（`r=2`）持有私有图像。
- HE 使用 Microsoft SEAL 4.1.1 的 **BFV**，不是 CKKS，也不是仅明文定点计算。
- BFV 多项式次数为 `8192`，明文模数为 `536903681`，系数模数使用
  `CoeffModulus::BFVDefault(8192)`，SEAL 安全等级检查为 `tc128`。
- MPC 使用仓库 BOLT/SCI 中已有的两方定点协议；数值在 37 bit 环上表示，默认
  小数位为 12 bit。
- HE 线性层结束后，ALICE 给结果加随机掩码，BOB 只解密掩码后的密文，双方得到
  加法分享。MPC 算子结束后仍保持分享状态，不公开中间张量。
- 默认只向 BOB 公开最终类别。只有显式设置 `DEBUG_LOGITS=1` 时才额外向 BOB
  公开 logits，该选项只能用于正确性调试。

各算子的协议分工如下：

| ViT 计算 | 使用方法 | 复用的已有路径 |
| --- | --- | --- |
| 图像切 patch | BOB 本地 | 只改变私有输入的布局，不产生公开值 |
| Patch Embedding | BFV HE | 密文输入乘 ALICE 的明文权重 |
| CLS token、位置编码、偏置、残差 | 加法分享上的本地加法 | ALICE 把私有常量加入自己的分享 |
| LayerNorm（均值、方差、倒平方根、gamma、beta） | SCI MPC | `FPMath::layer_norm_iron` |
| Q、K、V 投影 | BFV HE | 与 IRON BERT 线性层相同的密文×明文权重范式 |
| `QK^T / sqrt(head_dim)` | BFV HE ct-ct | `bolt_bert_compat_ct_ct_qk`，复用现有 BERT cross layout |
| Softmax | SCI MPC | `FPMath::softmax_fix_iron_1`，即 IRON 的 max、查表 exp 和安全除法路径 |
| `Softmax × V` | BFV HE ct-ct | `bolt_bert_compat_ct_ct_softmax_v` |
| Attention 输出投影 | BFV HE | 密文×明文权重 |
| FFN 第一、第二线性层 | BFV HE | 密文×明文权重 |
| GELU | SCI MPC | `FPMath::gelu_iron` |
| 最终 LayerNorm | SCI MPC | `FPMath::layer_norm_iron` |
| 分类器 | BFV HE | CLS token 的密文×明文权重 |
| Argmax | SCI MPC | 安全比较与选择，最终类别只向 BOB 重构 |

本实现没有使用 CKKS。工作区中 BLB 相关代码出现的 CKKS 与本 IRON ViT 目标无关。

## 2. 当前端到端模型

烟雾模型使用标准 pre-LN ViT 顺序：

```text
4x4x3 图像 -> 2x2 Patch Embedding -> CLS/Position
-> [LN -> MHA -> Residual -> LN -> FFN/GELU -> Residual] x 1
-> Final LN -> 3 类分类器 -> Secure Argmax
```

具体尺寸为 5 个 token、hidden dimension 8、2 个 head、FFN dimension 16、
1 个 Transformer block。fixture 生成器还会执行独立明文参考，写出期望类别和
参考 logits；明文参考不参与密态计算。

## 3. 依赖与构建

以下命令从同时包含 `IRON/` 和 `BOLT/` 的工作区根目录开始：

```bash
cd /home/dm/mind/MIND_code/Pencil
```

Ubuntu/Debian 安装构建依赖：

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake make libgmp-dev libssl-dev
```

确认 SCI 的 SEAL 和 Eigen 源码完整：

```bash
test -f BOLT/SCI/extern/SEAL/CMakeLists.txt
test -f BOLT/SCI/extern/eigen/CMakeLists.txt
```

构建 IRON ViT：

```bash
cd IRON
JOBS=4 bash scripts/build_secure_vit.sh
```

脚本在 `IRON/build/secure_vit/` 中构建 ViT 适配层，并复用仓库中已验证的 BLB/SCI
BFV、MPC 和转换静态库；原 BERT 的 SCI build 不会被覆盖。最终标准 B/16 二进制为：

```text
IRON/build/secure_vit/bin/IRON_VIT
```

如果 GMP 位于当前 Conda 环境，构建脚本会自动检测 `$CONDA_PREFIX/include/gmp.h`
和对应动态库。其他非标准安装位置可显式传入：

```bash
bash scripts/build_secure_vit.sh \
  -DGMP_INCLUDE_DIR=/path/to/include \
  -DGMP_LIBRARIES=/path/to/libgmp.so \
  -DGMPXX_LIBRARIES=/path/to/libgmpxx.so
```

## 4. 单机端到端运行

一键启动 ALICE 和 BOB：

```bash
cd /home/dm/mind/MIND_code/Pencil/IRON
bash scripts/run_secure_vit_e2e.sh
```

首次运行会自动生成确定性 fixture。成功时脚本会把密态预测与明文期望类别比较，
并生成：

```text
fixtures/vit_smoke/alice_model.txt
fixtures/vit_smoke/bob_image.txt
fixtures/vit_smoke/expected_prediction.txt
fixtures/vit_smoke/reference_logits.txt
outputs/vit_smoke/alice.log
outputs/vit_smoke/bob.log
outputs/vit_smoke/prediction.txt
outputs/vit_smoke/PASS
```

看到下面两行即表示完整协议路径通过：

```text
PREDICTION class=1
PASS: .../outputs/vit_smoke/prediction.txt
```

如端口冲突，可以更换端口和输出目录：

```bash
PORT=47200 OUTPUT="$PWD/outputs/vit_smoke_47200" \
  bash scripts/run_secure_vit_e2e.sh
```

仅在调试时公开 logits：

```bash
DEBUG_LOGITS=1 PORT=47220 OUTPUT="$PWD/outputs/vit_debug" \
  bash scripts/run_secure_vit_e2e.sh
```

## 5. 两终端或两台机器运行

先生成 fixture：

```bash
cd /home/dm/mind/MIND_code/Pencil/IRON
python3 ../BOLT/scripts/generate_vit_fixture.py \
  --protocol IRON --out-dir fixtures/vit_smoke
```

tiny 兼容 smoke 的两终端运行使用 `IRON_VIT_GENERIC`：

```bash
build/secure_vit/bin/IRON_VIT_GENERIC \
  r=1 p=47100 \
  model=fixtures/vit_smoke/alice_model.txt \
  output=outputs/vit_manual/prediction.txt
```

终端二启动持有图像的 BOB：

```bash
build/secure_vit/bin/IRON_VIT_GENERIC \
  r=2 p=47100 ip=127.0.0.1 \
  input=fixtures/vit_smoke/bob_image.txt \
  output=outputs/vit_manual/prediction.txt
```

跨机器运行时，把 BOB 的 `ip` 改为 ALICE 地址并放通端口。ALICE 不需要图像文件，
BOB 不需要模型文件；输出目录需提前创建。

## 6. 本仓库验证结果

真实 B/16 checkpoint 的当前实测结果（CPU SEAL，单机 loopback，两方）：

```text
VIT_FLOAT_TOP1=88
VIT_FIXED_TOP1=88
PREDICTION class=88
VIT_SECURE_TOP1_MATCH=PASS
PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0
VIT_HARDENING_TRACE_GATE=PASS blocks=12 rows=123
```

full-12 生产路径的实际 accounting：online `3954100.950 ms`，端到端
`3992559.222 ms`，双向通信 `239611653094 bytes`，logical rounds `23429`。
其中 breakdown 为 HE `898140.105 ms`、MPC `2144568.297 ms`、conversion
`910414.629 ms`，其余 local/runtime overhead `977.919 ms`；对应通信分别为
HE `0`、MPC `185082986510`、conversion `54488288418` bytes。

block-1 aggregate-only observer（只公开标量，不恢复中间 tensor）的三路最大绝对误差为：

| stage | clear/fixed | clear/secure | fixed/secure |
| --- | ---: | ---: | ---: |
| patch projection | 0.0071098276 | 0.0071105957 | 0 |
| attention residual | 0.1284458227 | 0.1284446716 | 0 |
| MLP residual | 1.5047338357 | 1.5047340393 | 0 |
| final CLS | 0.4332780852 | 0.4332771301 | 0 |
| logits | 0.2458925920 | 0.2458915710 | 0 |

完整的每个 operator shape、scale、ell/bit length、packing 和计时见
`outputs/vit_b16_full_current/operator_trace.csv`；完整 12 层的 27-stage
aggregate-only 误差见 `outputs/vit_b16_full_aggregate/secure_aggregate_errors.csv`
（该文件只包含 secure/clear 与 secure/fixed 的误差标量；对应的
`reference_report.json` 还包含每个 patch、attention residual、MLP residual、CLS
和 logits 阶段的 clear/fixed/secure 三路聚合误差）。

验收摘要由以下命令生成，且会对 checkpoint SHA、BFV 范围、CSV accounting、27 个
aggregate stage 和中间明文恢复约束 fail-closed 校验：

```bash
BOLT/.venv-vit-b16/bin/python BLB/vit_b16/scripts/summarize_vit_acceptance.py \
  --pytorch-reference BOLT/outputs/vit_b16_current/pytorch_reference.json \
  --full-reference IRON/outputs/vit_b16_full_reference_trace/reference_report.json \
  --bfv-audit IRON/outputs/vit_b16_full_reference_trace/bfv_range_audit.json \
  --block1-reference IRON/outputs/vit_b16_block1_aggregate/reference_report.json \
  --block1-performance IRON/outputs/vit_b16_block1_aggregate/performance.csv \
  --block1-breakdown IRON/outputs/vit_b16_block1_aggregate/backend_breakdown.csv \
  --seed-diagnostics IRON/outputs/vit_seed_regression_final/reports/seed_diagnostics.json \
  --imagenet50k-report IRON/outputs/vit_b16_imagenet_pytorch_50k.json \
  --full12-secure-dir IRON/outputs/vit_b16_full_aggregate \
  --output IRON/outputs/vit_b16_acceptance_summary.json
```

真实图片的 PyTorch/reference、fixed 和 secure 单图 top-1 均为 88；3-seed tiny
compatibility regression 的 secure/fixed 一致率为 3/3。ImageNet 的 50k 结果只属于
PyTorch/reference 浮点路径；secure 多图片 accuracy 明确为 NOT_RUN。

## 7. 代码位置与对比实验边界

- `BLB/vit_b16/blb_vit_protocol.cpp` 与 `BLB/vit_b16/BOLTTransformer/bolt_vit_compat_*`：
  标准 B/16 的 compact packing/执行图适配层。
- `BLB/vit_b16/vit_model.{h,cpp}`：真实 checkpoint loader；同时接受 BOLT/IRON wire
  magic，converter 支持 fused 或 independent Q/K/V。
- `BOLT/SCI/tests/vit_bolt/bolt_vit_protocol.cpp`：通用 ViT/原 IRON smoke 图，保持
  tiny fixture 与 BERT 兼容回归。
- `BOLT/SCI/src/FloatingPoint/fp-math.cpp`：原有的 IRON MPC 非线性算子实现。
- `BOLT/SCI/tests/bert_iron/`：本次复用方法的 IRON BERT 基线来源。
- `scripts/build_secure_vit.sh`：IRON 目标的隔离构建入口。
- `scripts/run_vit_b16_e2e.sh`：标准真实 checkpoint 的单机两方端到端运行与 gate。
- `scripts/run_vit_seed_regression.sh`：3-seed tiny compatibility regression。

标准 B/16 使用 BERT-compatible compact layout（T=256、D=768、H=12），不是逐标量
fallback；只新增模型适配/packing/loader/执行图和测试入口，BERT 的 BFV、MPC、OT、
HE↔MPC、Softmax、GELU、LayerNorm、MatMul、BSGS、key switching 与安全模型未修改。
