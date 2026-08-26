# BOLT ViT-B/16 审计、实现映射与验收状态

本文记录当前工作区快照中的 ViT 适配。它补充并更新早期的
`bolt_vit_gap_analysis.md`；早期文档描述的是未接入共享 BOLT packed backend
之前的旧 scalar/generic 草稿，不能用于判断当前 `BOLT_VIT` 二进制。

## 1. 先审计：BERT 专用硬编码

BERT 原路径仍保留原有约束，ViT 不直接改写这些文件：

| 文件 | BERT 专用内容 | 处理方式 |
|---|---|---|
| `SCI/tests/bert_bolt/linear.h` | `INPUT_DIM=128`、`COMMON_DIM=768`、`INTER_DIM=3072`、`ATTENTION_LAYERS=12`、`PACKING_NUM=12` | 仅作为 BERT 兼容入口保留；ViT 不引用其固定形状宏 |
| `SCI/tests/bert_bolt/bert.cpp` | 多处 BERT 12 层调度、序列/投影布局和 BERT-specific conversion | 不修改；ViT 走 `BOLTTransformer` 泛化接口 |
| `SCI/tests/bert_bolt/bert_utils.cpp` | 12 heads、每 head 64、768/3072 向量和 12 个 block 的模型检查 | 不修改；ViT checkpoint 由新的 strict loader 自己检查 197/768/12/3072 |
| `SCI/tests/bert_bolt/bert.h`、`nonlinear.cpp` | BERT 的 `NL_SCALE=12`、BERT layer API 和 GELU/Softmax 调度 | 不修改；ViT 复用 `FPMath` 的数学入口，不复制 BERT Add&Norm 图 |
| `SCI/src/BOLTTransformer/*_compat` | 为已部署 BERT layout 保留的兼容函数 | 不改数学定义；ViT 使用同一目录中新的运行时 shape metadata 和通用入口 |

当前 ViT 图只把 `tokens`、`physical_tokens`、`model_dim`、`heads`、`ff_dim`、
`layers` 和 `classes` 从 checkpoint/config 带入运行时。没有把 BERT 的 12 层、
128 token 或 768/3072 分支复制到 ViT 图中。

## 2. 可以直接复用的 BOLT 算子入口

| ViT 算子 | 入口 | 密码后端 |
|---|---|---|
| Patch/QKV/输出/MLP/分类 ct-pt | `BoltLinear::cipher_plain_matmul_bsgs` | BFV BatchEncoder + compact repeated-row + BSGS |
| QKᵀ | `BoltLinear::cipher_cipher_matmul_cross` | BFV ct-ct cross MatMul + relinearization |
| Softmax×V | `BoltLinear::softmax_v_matmul` | BFV ct-ct cross MatMul + relinearization |
| SS→HE / HE→SS | `BoltConversion::ss_to_he`、`he_to_ss` | 现有 BOLT packed conversion 和随机 mask |
| head slice/transpose/merge | `BoltAttentionPacking::slice_head`、`transpose_tokens`、`merge_heads` | 公开 rotation/mask 的 packed layout 操作 |
| Softmax | `FPMath::softmax_fix` | 原有 MPC Softmax |
| GELU | `FPMath::gelu_approx_2` | 原有 BOLT MPC GELU；ViT adapter 保留 ell20/scale11→scale4 调度 |
| LayerNorm statistics | `FPMath::layer_norm_fix` | 原有 MPC LayerNorm statistics |
| BFV setup | `BoltHE::BoltHE` | 现有 BFV context/key/relinearization/Galois key 路径 |

`BoltLinear`、`BoltConversion`、`BoltAttentionPacking` 的核心算法没有重写；本次
只增加/使用 ViT 运行时 shape 和物理 token metadata。

## 3. 当前 ViT secure 图映射

```text
BOB local CHW patchify
  → ct-pt BSGS patch projection
  → ALICE-private CLS/position share addition
  → public padding mask (197 valid / 256 physical)
  → for each pre-LN block:
       LN1 statistics + gamma/beta
       Q/K/V ct-pt BSGS
       QKᵀ ct-ct cross
       public padding mask, existing MPC Softmax on 197×197 valid rows
       Softmax×V ct-ct cross
       output projection ct-pt BSGS
       residual share add + public token mask
       LN2 → FC1 → GELU → FC2 → residual share add
  → final LN
  → public-index CLS extraction
  → 768→1000 ct-pt classifier
  → MPC argmax; only class index is revealed
```

ViT-B/16 的逻辑 token 数为 197，当前 packed physical row count 为 256。padding
处理有两个明确边界：

1. token mask 在 embedding、LayerNorm 输入/输出和 residual 边界清零无效 token；
2. QKᵀ 后在进入现有 `FPMath::softmax_fix` 前只取公开确定的有效 197×197 子矩阵，
   再把 Softmax 结果放回 256×256 的 packed layout。无效 key 不会产生 exp 项，
   因而不会改变有效 query 的分母；无效 query 的 Softmax×V 行保持为零。

这不是秘密条件分支，也不是随机 mask；它是固定公开 layout mask，通过已有
`FixOp::mul(secret_share, public_mask)` 实现。trace 中分别记录为
`bolt_public_fix_mul_token_mask` 和
`bolt_public_fix_mul_attention_padding_mask`。

## 4. 新增/修改的适配层文件

主要修改：

- `SCI/src/BOLTTransformer/bolt_metadata.h`：增加 logical/physical token metadata
  和运行时 shape 校验；不改变 HE/MPC 算法。
- `SCI/tests/vit_bolt/bolt_vit_protocol.{h,cpp}`：256 physical packing、公开
  padding mask、pre-LN block 图和 BOLT 算子 trace。
- `SCI/tests/vit_bolt/bolt_vit.cpp`：由 checkpoint 计算运行时 metadata，并写出
  logical/physical shape、固定点、通信和阶段 breakdown。
- `SCI/tests/vit_bolt/vit_model.cpp`：strict V3 model loader，并拒绝 trailing data。
- `scripts/preprocess_vit_image.py`：兼容 canonical manifest 和 Hugging Face
  `size/image_mean/image_std` 字段。

新增：

- `scripts/convert_vit_checkpoint.py`：真实 checkpoint strict loader/converter，
  支持 `.safetensors`、受限 CPU `torch.save(state_dict)` 和 `.npz` 测试接口；
  支持 fused QKV 与独立 Q/K/V；缺键/错形状/非 pre-LN/未 pinned SHA256 均失败。
- `scripts/test_vit_checkpoint_loader.py`：仅测试 deterministic schema mapping；
  synthetic tensor 不计为 checkpoint integration。
- `scripts/vit_reference.py`：独立 floating reference 和 fixed-clear reference，
  输出每个算子 shape/scale/bit-length/packing、clear↔fixed 误差和 padding invariance。
- `scripts/vit_pytorch_reference.py`、`scripts/compare_vit_logits.py`：可选的真实
  PyTorch/Hugging Face checkpoint phase-A runner 和 canonical logits comparator；
  缺 PyTorch 或真实权重时 fail closed。
- `scripts/run_vit_b16_e2e.sh`：真实图片的两方运行包装器；模型只传给 ALICE，
  BOB 只接收预处理图片，强制 `debug_logits=0`。

## 5. 尚存的精确性边界

- BOLT 现有 `FPMath::layer_norm_fix` 没有显式 epsilon；checkpoint 的 epsilon 被
  loader 和 metadata 记录，但不修改核心 LayerNorm 数学实现。标准 ViT 的
  `1e-12` 必须在 B/C 验证中单独报告，不能默认为已严格复现。
- `vit_reference.py` 的 fixed-clear softmax 使用 clear simulation of the
  existing `exp4` approximation，除法仍是无泄露验证用的 clear 数值模拟；它不
  读取 secure 中间 share。
- 正式 secure 图不提供中间明文 observer。因而 secure 中间 clear/fixed 误差必须
  标记 `NOT_AVAILABLE`，不能通过 `debug_logits` 或逐层解密来“补齐”。

## 6. 当前实测结果（当前快照）

| 项目 | 结果 | 证据 |
|---|---|---|
| BOLT_VIT、BOLT_BERT、BERT shared backend、ViT kernels/attention/stem 构建 | PASS | `cmake --build SCI/build_vit ...` |
| loader fused/独立 QKV schema 单测 | PASS | `CHECKPOINT_LOADER_UNIT=PASS` |
| 真实 ViT-B/16 checkpoint 下载、SHA256 和 strict C++ loader | PASS | `checkpoints/vit_b16/manifest.json`；SHA256 `1cea0711…622ccb5d`；197/768/12/3072/12/1000 |
| 隔离 PyTorch reference 环境 | PASS | `.venv-vit-b16/`；`torch==2.7.1+cpu`、`transformers==4.40.0` |
| PyTorch vs canonical clear reference（公开 smoke 图片） | PASS | top-1 都为 88，最大绝对误差 `2.24342285707e-6` |
| 真实 B16 fixed-clear top-1 对齐（公开 smoke 图片） | FAIL | PyTorch/clear top-1 `88`，fixed-clear top-1 `815` |
| BERT data preflight、shared backend 回归 | PASS | `outputs/bert_shared_backend_after_vit/` |
| tiny 单层/多层 secure smoke | PASS | `outputs/vit_final_smoke_after_public_mask/`、`outputs/vit_multilayer_after_vit/` |
| tiny attention/stem/kernel protocol tests | PASS | `outputs/vit_attention_after_vit/`、`outputs/vit_stem_after_vit/`、`outputs/vit_bolt_kernels_after_vit/` |
| tiny 10-seed secure final-class consistency | FAIL：9/10；seed `20260814` 不一致 | `outputs/vit_seed_regression_after_vit.csv` |
| fixed-clear max absolute error（上述 tiny seeds） | 已记录，`0.243425084079` | `outputs/vit_seed_diagnostics_after_vit/seed_diagnostics.json` |
| secure max absolute error、首个不一致层 | NOT_AVAILABLE | 正式路径没有 intermediate reconstruction observer |
| 完整真实 B16 secure inference | NOT_RUN | 已准备权重，但尚未启动高成本 12 层两方运行 |
| ImageNet-1k validation 数据与标签 | PASS | 官方 tar MD5 `29b22e2961454d5413ddabcf34fc5622`；50,000 张；devkit synset ID 经 `meta.mat` 映射到标准 WNID 类别索引 |
| ImageNet-1k PyTorch smoke accuracy | PASS（仅 16 张链路烟测） | top-1 `13/16 = 81.25%`；top-5 `16/16 = 100%`；`outputs/accuracy/vit_b16_pytorch_smoke16.json` |
| ImageNet-1k 全量 validation accuracy | NOT_RUN | 数据已具备；需运行 `scripts/evaluate_vit_b16_pytorch.py` 的 50,000 张全量命令后才能报告 |
| 真实 B16 两方 runner 启动链路 | PASS | checkpoint load、loopback listen/connect、BFV setup、MPC setup、online start 和 30s heartbeat 均实测；限时探针未冒充完整推理 |

因此当前总状态仍是 `FAIL/INCOMPLETE`，不是最终 ViT-B/16 验收 PASS；这不会覆盖
BERT 原路径的 PASS 结果。
