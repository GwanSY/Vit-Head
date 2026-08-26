# BOLT Transformer 共享后端依赖分析

本文档是 `SCI/tests/bert_bolt` 向 `SCI/src/BOLTTransformer` 抽取前的代码依赖边界。分析以实际 SEAL/SCI 调用为准，不以函数名或日志为准。

## 1. 文件与依赖

| 文件 | 密码实现/模型职责 | 共享化处理 |
|---|---|---|
| `he.h`, `he.cpp` | BFV parameters/context，`BatchEncoder`，public/secret/Galois/relinearization keys，SEAL encrypt/decrypt/evaluator | context/key/encoder/evaluator 进入 `bolt_he.*`；参数由 metadata 传入 |
| `linear.h`, `linear.cpp` | compact/cross packing，ct-pt BSGS，QK ct-ct，Softmax×V ct-ct，flooding/mod-switch | 内核进入 `bolt_linear.*`；BERT 尺寸不得进入共享函数 |
| `bert.cpp` | SS↔HE，prime-field↔`Z2^ell`，层级调度与 BERT 计算图 | SS↔HE 进入 `bolt_conversion.*`；层号、scale schedule 和模型图留在 BERT |
| `nonlinear.cpp` | `softmax_fix`、`gelu_approx_2`、`layer_norm_fix` 的 MPC wrapper | 数学定义和 MPC 实现不迁移、不修改 |
| `bert_utils.cpp` | 12 层/768 维等 BERT 参数，模型文件名和读取 | 全部留在 BERT，不进入密码后端 |
| `utils-HE.*` | key serialization helper，ciphertext send/recv，noise flooding | 共享后端复用这些现有 helper，不重写 SEAL 原语 |

## 2. 通用 BFV 逻辑

`HE::HE` 建立 `scheme_type::bfv` context，并保持以下对象：

- `SEALContext`、`Encryptor`、`Decryptor`、`Evaluator`；
- `BatchEncoder`；
- public key、Galois keys、relinearization keys；
- BOB 持有 secret key，ALICE 使用公开评估密钥执行 HE 计算。

这些对象与模型无关，进入 `bolt_he.*`。共享后端必须在运行前检查 `BatchEncoder`、Galois key 和 relinearization key，缺少时立即失败。

## 3. SS-to-HE 和 HE-to-SS

`bert.cpp` 的实际协议是：

1. SS-to-HE：BOB 将自己的 share 按 plaintext prime 映射、`BatchEncoder::encode`、encrypt 后发给 ALICE；ALICE 将本地 share 编码为 plaintext 并 `add_plain_inplace`。
2. HE-to-SS：ALICE 从 plaintext prime 中采样随机 mask，从 ciphertext 中 `sub_plain_inplace`，必要时加入 `p/2`，再发给 BOB；BOB decrypt/decode 只获得 masked share。
3. prime plaintext modulus 与 `Z2^ell` 之间的校正继续使用现有 `gt_p_sub`/shift 调度，不将 prime residue 直接解释成环上中间明文。

共享部分进入 `bolt_conversion.*`；不得为了转换或测试在中间边界重构完整激活。

## 4. ct-pt compact packing 与 BSGS

`Linear::bert_cipher_plain_bsgs` 和 `Linear::bert_cipher_plain_bsgs_2` 的通用核心是：

- 使用 BatchEncoder slots 保存 compact-packed 输入/权重对角线；
- 使用 `rotate_rows`/`rotate_columns` 构造 baby/giant steps；
- 使用 `Evaluator::multiply_plain` 计算 ct-pt 乘法；
- 加和后执行原 BOLT 配置的 noise flooding 和 modulus switching。

BERT 中 `n1/n2`、input/output/intermediate dimension 的选择逻辑改由 `BoltMatMulMetadata` 传入；后端不得再根据 768/3072 分支。

## 5. ct-ct cross packing

QKᵀ 的实际路径是 `bert_cipher_cipher_cross_packing`：对 compact/cross-packed Q/K 进行 `Evaluator::multiply`，随后 `relinearize_inplace`、rotation、mask 与累加。这一核心进入 `bolt_linear.*`，packing shape 及 baby/giant parameters 进入 metadata。共享接口不允许转发到 `LinearOT::matrix_multiplication`。

## 6. Softmax×V ct-ct

`Linear::softmax_v` 消费 `preprocess_softmax_s1_ct_ct` 产生的 Softmax ciphertext packing 和 `linear_1` 保留的 V ciphertext，执行 `Evaluator::multiply`、relinearization 和 modulus switching。该内核独立暴露为 `softmax_v_matmul`，不经过通用 MPC MatMul。

## 7. 必须留在 BERT 的模型逻辑

以下内容不属于共享密码后端：

- 12 层、128 token、768 hidden、3072 FFN、12 heads 等 BERT 固定尺寸；
- `bert-base-uncased` 参数文件路径、层号和文件读取；
- BERT 每层的 scale/bitwidth schedule；
- pooler/classifier 图、word elimination 和性能日志格式；
- Softmax、GELU、LayerNorm 的 MPC 数学定义。

## 8. 迁移不变式

1. BFV + prime plaintext modulus，禁止以 `2^ell` 替代 prime-field conversion。
2. 必须使用 `BatchEncoder`；禁止一个标量一个 ciphertext。
3. ct-pt 必须实际调用 `multiply_plain`、rotation 与 BSGS。
4. QKᵀ 和 Softmax×V 必须实际调用 `Evaluator::multiply` 与 relinearization。
5. 保留 flooding/mod-switch，且由 metadata 明确选择原 BOLT 参数。
6. 任何 key、packing 或 layout 前置条件不满足时 fail closed，禁止静默 fallback。
