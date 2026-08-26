# BLB-ViT faithful baseline

## 结论

canonical secure 入口是
`SCI/tests/bert_large_bolt/ckks_vit_b16_main.cpp`。旧的 `vit_b16` BOLT/BFV
兼容 target、独立 `vit_smoke_e2e` 和自定义 SEAL/GC/MPC adapter 已删除，不能再
作为 ViT 运行路径或性能基线。

ViT driver 使用 BERT BLB 的 `Linear_ckks`、Phantom CKKS、BSGS/cross-packing、
原始 `ckks_to_mpc`/`mpc_to_ckks`、Softmax、LayerNorm 和 GELU。ViT-specific 代码
只处理 checkpoint/layout、patch embedding、position/CLS embedding、pre-norm
residual boundary 和 CLS classifier。

## 路径审计

| 模块 | BERT BLB 实现 | 当前 ViT 实现 | 是否一致 |
|---|---|---|---|
| Q/K/V projection | `linear_1` + 原始 CKKS ct-pt/BSGS | ViT Q/K/V 转为 BERT packing 后调用同一入口 | 是 |
| QK^T | 原始 CKKS ct-ct cross-packing | 同一 kernel，物理长度为 256 | 是 |
| Softmax | 原始 BLB Softmax | 同一 Softmax；公开 mask 只屏蔽 padding key | 是 |
| Softmax×V | 原始 CKKS packing/ct-ct 路径 | 同一 `Softmax_v_linear_layer_norm_part1` | 是 |
| Output projection | 原始 `linear_2`/BSGS2 | 同一 `linear_2`/BSGS2 | 是 |
| LayerNorm | BLB MPC square/mean/rsqrt + CKKS affine | 同一原语，在 ViT pre-norm 边界调用 | 是 |
| FFN1 | 原始 BSGS2 | 同一 BSGS2 | 是 |
| GELU | 原始 CKKS polynomial + MPC GELU | 同一 polynomial + MPC GELU | 是 |
| FFN2 | 原始 BSGS2 | 同一 BSGS2 + ViT residual 边界 | 是 |
| pooling/classifier | BERT pooler/classifier | ViT CLS-only 1000-class head；模型结构不同 | 模型边界不同 |
| CKKS↔MPC conversion | 原始 `ckks_to_mpc`/`mpc_to_ckks` | 同一 conversion；次数随 ViT 图边界变化 | 是 |

重点检查：

- packing：复用 BERT feature-major、cross-packing、BSGS2 split-half layout；
- Phantom：复用 `HE_CKKS(..., LOCATION::DEVICE)`，没有 BFV BatchEncoder；
- MHP/BSGS：QKV、QK、Softmax×V 和 FFN linear 全部来自 `linear_ckks.cpp`；
- conversion：没有独立 RNS import、persistent plaintext cache 或额外转换协议；
- token expansion：197 逻辑 token 扩为 256 物理 token，是原始 full-slot packing
  的限制，不是截断优化；padding query 仍执行完整 BLB 计算；
- truncation：ViT main 在进入协议前启用现有 `local_truncation_mode`，与 12 层
  BERT 使用相同开关，不新增截断算法；
- preprocessing：构造 `Bert_ckks` 时一次性执行原始全层 preprocessing，不做
  ViT 专用按层 streaming 或释放缓存。

## 复杂度差异（理论工作量）

ViT-B/16 与 BERT-base 均为 `D=768`、`H=12`、`FF=3072`。因此 FFN hidden
dimension 和 head width 没有变化；主要差异来自物理 token 数和 ViT 模型边界。

| 因素 | BERT | ViT logical | ViT physical | 相对 BERT（physical） |
|---|---:|---:|---:|---:|
| token-linear work | 128 | 197 | 256 | 2.000× |
| QK attention cells | 128² | 197² | 256² | 4.000× |
| Softmax rows/elements | 12×128² | 12×197² | 12×256² | 4.000× |
| FFN hidden dimension | 3072 | 3072 | 3072 | 1.000× |
| extra model boundaries | 无 | patch/embedding/final LN/CLS head | 同左 | ViT-only |

因此不能把旧 BFV/BOLT run 的 HE/MPC/conversion 秒数直接归因于原始 BLB。
实际比例必须从同一 backend、checkpoint、token、security 参数和网络设置下的
profile CSV 计算；baseline 不输出伪造的 wall-clock 贡献百分比。

## Profiling 输出

每个 Transformer block 写入 `block_profile_party{1,2}.csv`：

- HE：`linear1_he_ms`、`linear2_softmax_v_he_ms`、`linear3_he_ms`、
  `linear4_he_ms`、`he_time_ms`、ct-pt/ct-ct multiply、rotation、rescale；
- MPC：Softmax、LayerNorm、GELU 时间和 MPC communication bytes；
- conversion：CKKS→MPC/MPC→CKKS 次数、ciphertext 数、conversion bytes；
- communication：ciphertext transfer、总 bytes 和 rounds。

`communication_breakdown_party{1,2}.csv` 记录 patch embedding、position/CLS
embedding、每个 block 的 LayerNorm/QKV/Softmax/Softmax×V/FFN、final LayerNorm、
classification、initialization 和 total。运行后用
`vit_b16/scripts/summarize_blb_vit_baseline.py` 生成 Alice+Bob 双向合计。

## 执行流程

```text
BOB image -> public patchify -> BLB MPC input
  -> original BLB CKKS BSGS2 patch projection
  -> Alice CLS/position shares
  -> 12 x [BLB pre-LN -> QKV/QK^T -> Softmax
           -> Softmax*V/output projection -> residual
           -> BLB pre-LN -> FFN1/GELU/FFN2 -> residual]
  -> BLB final LayerNorm -> CLS -> classifier -> public prediction
```

## BLB 原方法限制

原始 BERT packing 没有 197-token 专用布局，故使用 256 物理 token；没有设计
197 专用 packing。BERT pooler 也不能移植到没有该层的 ViT checkpoint。若某个
BLB 组件不能直接覆盖 ViT 形状，标记为“BLB 原方法限制”，不在 baseline 中设计
替代协议。

## Hetero-fixed 比较入口

后续 Hetero-fixed 实验应固定本 baseline 的模型 checkpoint、197/256 token 设置、
安全参数、Phantom backend、GPU 可见性和网络配置，以
`ckks_vit_b16_main` 的 profiling 输出作为比较入口。任何新算子或新 conversion
必须在独立 target/分支中显式标注，不回写 BLB baseline。
