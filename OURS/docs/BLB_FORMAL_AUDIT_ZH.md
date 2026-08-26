# BLB 正式 BERT/ViT 路径审计

审计日期：2026-08-04  
审计根目录：`/home/dm/mind/MIND_code/Pencil/BLB`  
论文：*Breaking the Layer Barrier: Remodeling Private Transformer Inference with Hybrid CKKS and MPC*（USENIX Security 2025）

## 结论

当前仓库包含 BLB 的 BERT-base 算子骨架、五组 CKKS 参数、FineGrainFusion 函数、CKKS ct-ct/ct-pt MatMul、多头打包、BSGS 和 CKKS/MPC 转换代码。正式源码中的全 1 QKV/FFN 权重、fake V、编码零向量转换份额和自相加残差已经从活动路径移除，但当前提交仍不是可验收的正式端到端结果。

构建状态与复现状态分开记录：目录内 GPU/Phantom 目标已经成功构建为 `build_blb_formal_oracle_gpu/sci/bin/ckks_bert_large_main`，本轮构建日志为 `outputs/build_formal_oracle_gpu_20260805.log`，并明确标记 `formal_ready=no`。三 oracle 的 BLB-local 参考份额、manifest 校验和安全标量 MSE 入口已实现；尚未在官方容器中完成五块两方实测，因此不能据此声称单 block、BERT-base 或 ViT 已复现。

必须区分两条路径：

- `vit_adapter/` 是 tiny ViT 协议链路测试，正式名称为 `vit_smoke_e2e`。其 QK、Softmax×V、投影、LayerNorm、FFN、pooling 和分类头使用 GC/MPC，不符合论文 BLB 的正式协议划分，任何 PASS 都不得作为论文结果。
- `SCI/tests/bert_large_bolt/` 是论文 BLB/FineGrainFusion 正式路径。官方容器中的真实 12 层 MRPC BERT-base checkpoint 和输入已经复制到 BLB 并通过 manifest；第 0 层按 `bert.py` 的 post-norm 顺序接入真实输入、两组 LayerNorm 参数、attention/FFN residual 和三 oracle 比较入口，当前主要阻塞是官方容器中的五块 MSE 尚未实测。

## 目录边界

允许读取、修改、构建和生成结果的唯一项目根目录是：

```text
/home/dm/mind/MIND_code/Pencil/BLB
```

允许使用操作系统提供的编译器、CMake、OpenSSL、GMP、OpenMP、pthread、CUDA runtime 等系统依赖，但禁止链接相邻工程的源码、库、脚本和构建产物。当前验收使用 `BLB/build_blb_formal_oracle_gpu/`；其他 `build_blb_*` 目录只是 BLB 内独立构建缓存，原有 `SCI/build/` 是迁移前遗留构建，动态依赖不完整，不得用于正式运行。

原正式 BERT 源码的越界模型路径已经移除。`ckks_bert_large_main` 通过 `model=` 接收权重目录，`repro/run_original_bert.sh` 强制该目录位于 `BLB` 内，默认是 `BLB/checkpoints/official-blb-mrpc/weights_txt`；旧 `/ezpc_dir` 调试输出也已迁到 `BLB/outputs/legacy_debug`。该目录的 12 层 checkpoint manifest 已通过；正式运行会先验证 BLB-local `oracle_shares/manifest.json`，缺失或哈希/shape 不符时在联网前阻塞。

## 算子协议域审计

下表的“正式域”来自论文 Table 2、Figure 10 和仓库已有 BLB 代码；不是对 `vit_smoke_e2e` 当前行为的描述。

| Transformer 算子 | 正式协议域 | 仓库实现入口 | 打包/融合说明 | 当前状态 |
|---|---|---|---|---|
| residual / element-wise add | CKKS | `ckks_bert.cpp` 各融合函数中的 `add_inplace` / `add_plain_inplace` | 融入相邻块，不在 MPC 层间重构 | 骨架存在 |
| LayerNorm 最终仿射收尾 | CKKS ct-pt/ct-ct | `layer_norm_affine_ckks`（`he4`） | 复用 Figure 10 Block 1 的 host packing | 已接真实 ln2 `w/b`；安全实测待容器 |
| LayerNorm: sum、均值、中心化、平方、仿射 | CKKS | `linear4_add_layernorm_part1`、`layer_norm_part2_attention`、`Softmax_v_linear_layer_norm_part1`、`ln_part2_linear3_gelu_part1` | Reduction、Identity、Expansion 跨层融合 | 骨架存在 |
| LayerNorm: `rsqrt` | MPC/OT | `Bert_ckks::layer_norm_nonlinear` → `FPMath::sqrt(..., true)` | 复用 Bumblebee 非线性路径 | 存在；缺正式门禁 |
| Q/K/V projection | CKKS ct-pt MatMul | `Linear_ckks::linear_1` → `bert_cipher_plain_bsgs` | 多头交叉打包 + BSGS | 已接 `pp_1`；待 checkpoint 实测 |
| QK^T | CKKS ct-ct MatMul | `Linear_ckks::bert_cipher_cipher_cross_packing` | 多头批打包、交叉 packing、旋转复用 | 路径存在；待 oracle 实测 |
| Softmax: max/比较 | MPC/OT `cmp`、`mux` | `softmax_he` → `softmax_fix_he` → `FPMath::max` | Bumblebee 路径 | 存在 |
| Softmax: 减法、exp 多项式、sum、乘法 | CKKS | `exp_he`、`element_wise_mul` 及 `softmax_fix_he` | Figure 10 Block 3 | 存在；转换安全性未闭合 |
| Softmax: reciprocal | MPC/OT `rec` | `div_batch_he_V2` 及 `FPMath` reciprocal/sqrt 路径 | Bumblebee 路径 | 入口存在；需逐阶段对齐 |
| Softmax×V | CKKS ct-ct MatMul | `Linear_ckks::softmax_v` / `bert_softmax_V` | 多头 packing；不得改为逐元素 MPC | fake V 已移除；真实 V/input residual 已接入，待 oracle 实测 |
| attention output projection | CKKS ct-pt MatMul | `Linear_ckks::linear_2` → `bert_cipher_plain_bsgs_2` | BSGS | 已接 `pp_2`；待 checkpoint 实测 |
| FFN1 / FFN2 | CKKS ct-pt MatMul | `ln_part2_linear3_gelu_part1`、`linear4_add_layernorm_part1` → `linear_2` | 分别融合进 Block 5 和 Block 1 | 已接 `pp_3/pp_4`；FFN2 residual 是 attention 后真实 ln1 仿射输出，并通过 CKKS→MPC 安全缓存 |
| GeLU 线性多项式部分 | CKKS | `eval_gelu_poly`、`ln_part2_linear3_gelu_part1` | Figure 10 Block 5 | 存在 |
| GeLU 比较和选择 | MPC/OT `cmp`、`mux` | `gelu_nonlinear` → `MSB`、`LT`、`if_else` | 复用 BOLT GeLU 路径 | 存在 |
| pooling | CKKS | 正式路径应使用 `BertModel::w_p/b_p` | 论文要求 HE 线性算子 | 模型字段存在，CKKS 正式调用未接入 |
| classifier | CKKS | 正式路径应使用 `BertModel::w_c/b_c` | 论文要求 HE 线性算子 | 模型字段存在，CKKS 正式调用未接入 |

## Figure 10 五个 FineGrainFusion 块

仓库的五套 HE 实例与论文 Table 6 一一对应。源码中的 modulus 个别参数与论文表格相差 1 bit，正式复现前必须以论文表格和 MSE `< 1e-11` 双重门禁确认，不能仅凭进程退出码确认。

| 块 | Figure 10 融合范围 | 主要入口 | 论文参数（Degree / coeff modulus / scale） | 当前状态 |
|---|---|---|---|---|
| Block 1 | FFN2 → residual add → 下一 LayerNorm 的线性前半段，直到 `rsqrt` | `linear4_add_layernorm_part1`，`he4` | 16384 / `{60,60,60,60,60}` / 38 | real residual/安全 mask 代码已接；MSE 未实测 |
| Block 2 | post-norm layer 0 的真实 embedding → Q/K/V → QK^T；后续层复用 LayerNorm 后半段 | `attention_from_real_input`/`layer_norm_part2_attention`，`linear_1`，`bert_cipher_cipher_cross_packing`，`he1` | 32768 / `{60,35,60,60,60,60,60,60}` / 44 | `pp_1`/安全 mask 已接；MSE 未实测 |
| Block 3 | Softmax 中围绕 `cmp/mux/rec` 的 sadd/ewmul/sum/smul 线性算子 | `softmax_he`、`softmax_fix_he`、`exp_he`，`he_softmax` | 32768 / `{60,40,40,40,40,40,40,40,60}` / 40 | exp/ewmul 转换已改安全 mask；MSE 未实测 |
| Block 4 | Softmax×V → output projection → residual add → LayerNorm 线性前半段 | `Softmax_v_linear_layer_norm_part1`，`softmax_v`，`linear_2`，`he2` | 32768 / `{60,60,60,60,60,60,60}` / 43 | fake V 已移除；真实 layer-0 V/residual 已接入，BLOCKED：未产生 oracle MSE |
| Block 5 | LayerNorm 线性后半段 → FFN1 → GeLU 中围绕 `cmp/mux` 的线性多项式 | `ln_part2_linear3_gelu_part1`、`eval_gelu_poly`，`he3` | 32768 / `{60,32,60,60,60,60,60,60}` / 46 | `pp_3`/安全 mask 已接；MSE 未实测 |

## CKKS/MPC 转换边界

论文安全边界如下：

1. CKKS→MPC：服务端直接从多项式环 `A_N,q` 均匀采样 `r_hat`，对密文做 `Enc(Encode(x)) + r_hat`；客户端解密得到 `Encode(x)+r_hat` 的域份额，服务端持有 `-r_hat`。双方调用 field-to-ring，再各自在扩展环上执行本地 CKKS decode/FFT 和本地截断。
2. MPC→CKKS：双方对自己的环份额本地 CKKS encode，调用 ring-to-field；客户端加密自己的域份额并发给服务端，服务端用 plaintext addition 加入自己的份额。
3. 任一转换都不能重构 `x`，也不能把 `Encode(r)` 当作均匀多项式 mask。

仓库入口为：

- `Bert_ckks::ckks_to_mpc`：INTT → field-to-ring → 本地 decode；
- `Bert_ckks::mpc_to_ckks`：本地 encode → ring-to-field → NTT；
- `NonLinear::field_to_ring` / `ring_to_field`：模数转换；
- 各 FineGrainFusion 调用点负责密文掩码、传输、解密和份额拼接。

当前实现新增统一 `mask_ckks_to_mpc_inplace`：使用 SEAL 默认密码学安全随机工厂，在 ciphertext `parms_id` 对应的当前 RNS 模数链上调用 `sample_poly_uniform`，直接修改 `c0`，构造同 level/scale 的 `-r` Plaintext 份额。该随机量不经过 `CKKSEncoder`。`ckks_to_mpc` 随后调用真实 `NonLinear::field_to_ring`。代码级、checkpoint 和真实输入接线门禁 PASS，但两方 oracle 实测日志缺失，安全验收仍为 BLOCKED。

## 当前正式阻塞项

1. 官方 MRPC BERT-base 的 12 层权重和真实 128×768 输入已经落入 BLB 并通过 SHA256 manifest；checkpoint 缺失项解除。
2. 单 block 主程序已把该真实 layer-0 输入张量、两组 LayerNorm 参数和 attention input residual 接入 secure main；FFN2 使用 attention 后真实 ln1 仿射输出的安全份额，不再缓存旧的 h3 或模拟值。LayerNorm 已取消临时 `w/b`，缺真实参数份额直接抛错。三 oracle manifest 在构造网络前校验。
3. `repro/prepare_bert_oracle.py` 生成 FP32/fixed-clear 对照及 CSPRNG additive shares，`secure_stage_mse` 只通过 MPC 打开最终标量；五块尚未产生逐块 MSE `<1e-11` 实测值。
4. `ATTENTION_LAYERS` 按验收顺序保持 1；单层 PASS 前禁止扩成 metadata 的 12。
5. pooling/classifier 正式 CKKS 调用未接，故 `formal_ready=no`。
6. 统一计数入口已加入，但尚无完整两方运行产生的 rotation、ct-ct/ct-pt、转换、截断、通信和延迟有效数值。
7. 原 `SCI/build` 遗留二进制不得使用；正式运行必须显式指定本轮已验证的 `build_blb_formal_oracle_gpu/sci`（或重新在 BLB 内生成并记录一个等价的构建根）。

## 验收标准

正式结果必须同时满足：

- 审计脚本输出 `BLB_FORMAL_AUDIT status=PASS`；
- 五个融合块分别通过明文对照，CKKS 块 MSE `< 1e-11`；
- BERT 单 block、12-layer BERT-base、原结构 ViT 三层门禁依次通过；
- 日志包含每阶段 MSE、rotation、CKKS→MPC、MPC→CKKS、truncation、双方总通信量和端到端延迟；
- 日志证明 QK^T 和 Softmax×V 命中仓库 ct-ct、多头 packing、BSGS 路径；
- checkpoint、输入、构建目录、二进制和输出均位于 BLB 根目录；
- `vit_smoke_e2e` 的结果永远带 `formal_result=no`，不得进入正式汇总。
