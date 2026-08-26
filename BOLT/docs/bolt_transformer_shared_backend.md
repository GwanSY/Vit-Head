# BOLT Transformer 共享密码后端

## 1. 本阶段结论

本阶段完成了 BOLT BERT/ViT 可共享的 Transformer 密码内核，但**没有完成整个 ViT 计算图迁移**：

```text
BERT_SHARED_BACKEND=PASS
VIT_BOLT_KERNELS=PASS
FULL_VIT_MIGRATION=NOT_STARTED
```

`BOLT_VIT` 目标已链接 `SCI-BOLTTransformer`，但原 `run_vit` 仍包含 scalar BFV adapter 和 generic OT MatMul。本阶段按 fail-closed 要求使它立即报错，不会静默跑出一个非 BOLT 的结果。可运行的验证目标是 `BOLT_VIT_KERNEL_TEST`。

## 2. 密码方法

共享后端使用的 HE 是 Microsoft SEAL **BFV**，不是 CKKS，也不是把 plaintext modulus 设为 `2^ell` 的 scalar adapter。

- Transformer 线性层：BFV ciphertext-plaintext MatMul，`BatchEncoder` compact packing + rotations + BSGS + `multiply_plain`。
- QKᵀ：BFV ciphertext-ciphertext cross MatMul，实际调用 `Evaluator::multiply` 和 `relinearize_inplace`。
- Softmax×V：BFV ciphertext-ciphertext MatMul，实际调用 `Evaluator::multiply`、relinearization 和 BOLT modulus switching。
- HE↔MPC：使用 BOLT 现有的 prime plaintext modulus 与 `Z2^ell` additive share 语义。SS-to-HE 由 BOB 加密 share、ALICE `add_plain`；HE-to-SS 由 ALICE 减去 prime-field 随机 mask，BOB 只解密 masked share。`p/2`、`gt_p_sub` 和 shift 调度仍保留在原 BOLT 层级调度中。
- Softmax、GELU、LayerNorm：本阶段不修改数学定义，仍使用 SCI/BOLT 的 MPC 函数。

共享后端保留 public key、Galois keys、relinearization keys、noise flooding 和 modulus switching。`BoltHE` 中 secret key 只由 BOB 持有。

## 3. 目录与接口

```text
SCI/src/BOLTTransformer/
├── bolt_metadata.h       维度、packing、BSGS、flood/mod-switch metadata 和计数器
├── bolt_he.h/.cpp       BFV context、BatchEncoder 和评估密钥
├── bolt_conversion.h/.cpp  packed SS↔HE
└── bolt_linear.h/.cpp   ct-pt BSGS、QK ct-ct、Softmax×V ct-ct
```

主要接口：

```cpp
BoltCipherTensor BoltConversion::ss_to_he(...);
SecretSharedTensor BoltConversion::he_to_ss(...);

BoltCipherTensor BoltLinear::cipher_plain_matmul_bsgs(...);
BoltCipherTensor BoltLinear::cipher_cipher_matmul_cross(...);
BoltCipherTensor BoltLinear::softmax_v_matmul(...);
```

同时提供 BOLT BERT 已部署 full-slot layout 的 compatibility 入口：

- fused Q/K/V compact-packed ct-pt BSGS；
- 输出投影/FFN 的 compact-packed ct-pt BSGS；
- QKᵀ cross packing ct-ct；
- Softmax×V ct-ct；
- BERT 原 full-slot SS-to-HE/HE-to-SS wire layout。

BERT 的层号、checkpoint 文件读取、12/768/3072 模型配置和定点层级调度仍留在 `tests/bert_bolt`；共享内核只消费 metadata，不读模型文件。

## 4. fail-closed

以下情况都抛出异常，没有 fallback：

- 缺少 `BatchEncoder`、Galois key 或 relinearization key；
- metadata 表示 scalar ciphertext 或 stride 不能整除 BatchEncoder row；
- ct-ct 两个输入不在同一 modulus level；
- `BOLT_VIT` 调用旧 `BfvLinear` scalar adapter；
- QKᵀ 或 Softmax×V 调用 `LinearOT::matrix_multiplication`。

microbenchmark 的 `BOLT_FAIL_CLOSED ... status=PASS` 同时验证上述五个拒绝分支。

## 5. 构建

所有依赖、构建目录和输出都在当前 `BOLT/` 目录下。

```bash
cd BOLT

# 构建原 BERT，它现在链接并默认调用共享后端
TARGET=BOLT_BERT scripts/build_vit.sh

# 构建 BERT 原/共享内核回归
TARGET=BERT_SHARED_BACKEND_REGRESSION scripts/build_vit.sh

# 构建 ViT 三内核 microbenchmark
TARGET=BOLT_VIT_KERNEL_TEST scripts/build_vit.sh

# 仅验证目标可编译；运行会对未迁移 scalar/generic 路径 fail closed
TARGET=BOLT_VIT scripts/build_vit.sh
```

## 6. 运行回归

```bash
cd BOLT
scripts/run_bert_shared_backend_regression.sh
scripts/run_vit_bolt_kernels.sh
```

BERT 回归使用确定性 tiny BERT compact-packed fixture，分别运行抽取前原函数体和共享内核，覆盖 ct-pt BSGS、QK cross packing 和 Softmax×V。ct-ct 回归使用一个代表性 head-pair 控制测试内存，生产 BERT metadata 仍为 6 个 paired-head groups。密钥方只在测试最终边界解密，对所有输出 slots 做 hash/相等比较。它不需要 checkpoint，不是完整 BERT 数据集推理。

输出：

- `outputs/regression/bert_shared_backend.csv`
- `outputs/regression/bert_shared_backend_alice.log`
- `outputs/vit_bolt_kernels/alice.log`
- `outputs/vit_bolt_kernels/bob.log`

## 7. 当前实测结果

```text
BERT ct-pt BSGS:
original_output_hash=6dc7338deabdece0
shared_output_hash=6dc7338deabdece0
original/shared ciphertexts=1/1
original/shared rotations=4/4
status=PASS

BERT QK cross packing (representative head-pair):
original_output_hash=308d2269cca704ec
shared_output_hash=308d2269cca704ec
original/shared ciphertexts=1/1
original/shared rotations=896/896
status=PASS

BERT Softmax×V (representative head-pair):
original_output_hash=4c8472adb97fea81
shared_output_hash=4c8472adb97fea81
original/shared ciphertexts=1/1
original/shared rotations=128/128
status=PASS

ViT ct-pt:
rotations=4 multiply_plain=8 ciphertexts=2 status=PASS

ViT QK ct-ct:
rotations=22 evaluator_multiply=8 relinearizations=8 ciphertexts=2 status=PASS

ViT Softmax×V ct-ct:
rotations=22 evaluator_multiply=8 relinearizations=8 ciphertexts=2 status=PASS
```

## 8. 尚未迁移的 ViT 算子

本阶段未替换 `run_vit` 的完整图，因此以下 ViT 接线都尚未开始：

- patch projection；
- fused Q/K/V 投影与 ViT head packing；
- QKᵀ 在完整 attention 图中的接线；
- Softmax 前后 HE/MPC conversion 与 Softmax×V 接线；
- attention 输出投影；
- 两个 block LayerNorm 和 final LayerNorm 的 packed affine 接线；
- FC1、GELU 前后 scale schedule、FC2；
- classifier head；
- checkpoint 架构/pre-norm 参数的接入。

因此当前不存在可静默执行的 scalar/generic fallback：旧代码尚在，但 `BOLT_VIT_PROTOCOL_ENFORCEMENT` 会拒绝它。只有完成上述整图接线后，才能宣称完整 BOLT-ViT。
