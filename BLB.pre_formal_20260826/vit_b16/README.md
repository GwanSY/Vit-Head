# BLB ViT-B/16 baseline

ViT-B/16 的 secure baseline 只有一个入口：
`SCI/tests/bert_large_bolt/ckks_vit_b16_main.cpp`。它保留完整模型结构：
patch embedding、position/CLS embedding、12 个 pre-norm Transformer block、
final LayerNorm 和 1000 类 classification head。

执行图只调用 BERT BLB 已有的 CKKS 与 MPC 原语：

```text
patch embedding (BLB BSGS2 CKKS)
  -> position + CLS share
  -> 12 x [BLB LayerNorm MPC/CKKS
           -> BLB Q/K/V + QK^T CKKS
           -> BLB Softmax
           -> BLB Softmax*V + output projection CKKS
           -> BLB LayerNorm MPC/CKKS
           -> BLB FFN1 + GELU + FFN2]
  -> final LayerNorm
  -> CLS classification head
```

ViT 权重通过 `vit_model.{h,cpp}` 加载，并在 driver 中转换为 BERT
`BertModel` 的输入主序和多头 packing。物理序列长度固定为 256：197 个逻辑
token 加 59 个公开 padding 行，以满足原始 BLB full-slot packing；padding query
仍走原始计算，padding key 由公开 mask 屏蔽。

当前 baseline 明确保持：

- 原始 BLB CKKS ct-pt/ct-ct、BSGS、cross-packing 和 Phantom backend；
- 原始 BLB Softmax、LayerNorm、GELU 和 CKKS↔MPC conversion；
- 与 12 层 BERT 一致的现有 local truncation mode；
- 原始全层 weight preprocessing。ViT 不再按层 streaming，也没有 persistent
  plaintext cache、额外 RNS conversion、独立 BFV/BOLT 路径或 smoke executor。

ViT 的模型图适配只包含标准 attention 的 `1/sqrt(64)=1/8`：它在转换成
BLB 的 QKV 参数时折入 Q 的 weight/bias；BLB 的 QK^T evaluator 不增加额外
乘法。QKV 仍固定使用 weight `s=6`、bias `s=11`，全局 secure tensor 仍为
`ell=37, s=12`。

## 构建

从 BLB 根目录执行：

```bash
CUDA_ROOT=/usr/local/cuda-12.3 \
CUDA_ARCH=80 \
USE_HE_GPU=ON \
JOBS=2 \
BUILD_ROOT="$PWD/build_blb_formal" \
bash repro/build_formal_blb.sh
```

GPU 构建默认使用 VOLE。IKNP 仍保留，可显式构建 IKNP 版本：

```bash
USE_VOLE=OFF bash repro/build_formal_blb.sh
```

产物为 `build_blb_formal/sci/bin/ckks_vit_b16_main`。CPU 构建可将
`USE_HE_GPU=OFF`，但不会产生 Phantom GPU 运行结果。

## 运行与 profiling

需要先准备 canonical checkpoint 和 Bob 的图像 share，然后运行：

```bash
BLB_BUILD="$PWD/build_blb_formal/sci" \
BLB_DEVICE=2 \
BLB_MODEL="$PWD/checkpoints/vit_b16/bolt_model.txt" \
BLB_INPUT="$PWD/outputs/vit_b16_gpu/bob_image.txt" \
BLB_OUTPUT="$PWD/outputs/blb_vit_ckks_gpu2_lan" \
bash vit_b16/scripts/run_blb_ckks_baseline.sh
```

只测第一个 Transformer block（默认仍是完整 12 层）可设置
`BLB_STOP_AFTER_BLOCK=1`；该模式跳过 final LayerNorm/classifier，只输出已完成
block 的真实 secure latency 和通信 profiling。

每个 party 的输出包括：

- `latency_party*.csv`：secure end-to-end latency 和通信总量；
- `block_profile_party*.csv`：HE 时间、ct-pt/ct-ct multiply、rotation、rescale、
  BLB 非线性阶段（Softmax bucket、LayerNorm bucket、GELU strict core）时间、
  conversion 次数/bytes；LayerNorm bucket 明确包含 BLB 规定的 CKKS square/affine
  边界，不能误读为纯 MPC 时间；
- `communication_breakdown_party*.csv`：patch、embedding、每个 block、final LN、
  classification、initialization 等阶段；
- `summarize_blb_vit_baseline.py` 生成 Alice+Bob 双向合计。

## BLB 原方法限制

原始 BERT packing 没有 197-token 专用布局，因此 baseline 使用 256 个物理 token。
ViT 的 patch/position/CLS/classifier 是模型图边界，不是新的密码学协议；BERT
pooler 不能凭空移植到 ViT。后续 Hetero-fixed 方法应从同一
`ckks_vit_b16_main` 的模型、token、参数、backend 和网络配置进入，另行替换明确
标注的算子，不修改本 baseline。
