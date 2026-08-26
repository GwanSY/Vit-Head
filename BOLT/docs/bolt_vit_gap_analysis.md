# BOLT-ViT 协议保持差距审计

审计日期：2026-08-04  
审计对象：当前工作区中的 `BOLT` ViT 扩展快照（用户指定为 `client_opt`）  
审计方式：静态调用链追踪；本阶段未修改、替换或运行协议实现。

## 1. 结论

**总判定：FAIL。当前 `BOLT_VIT` 不是协议保持的 BOLT-ViT 适配。**

当前实现能够组成一个 BFV+MPC 的两方 ViT 烟雾测试，但关键密码后端与 BOLT 原路径不同：

1. Patch projection、Q/K/V、输出投影、FC1、FC2、LayerNorm affine 和分类头全部进入 `SCI/tests/vit_bolt/bolt_vit.cpp` 内自定义的 `BfvLinear::linear`。该函数逐标量加密、逐标量 `Evaluator::multiply_plain`，没有 `BatchEncoder`、compact/cross packing、旋转或 BSGS。
2. QKᵀ 和 Softmax×V 都进入 `secure_matmul`，再进入 SCI 的通用 `LinearOT::matrix_multiplication`。它们不是 BOLT 的 BFV ciphertext-ciphertext MatMul。
3. ViT 目标没有链接 `SCI/tests/bert_bolt/linear.cpp`、`he.cpp`、`bert.cpp`；因此不可能在运行时调用 `Linear::linear_1`、`Linear::linear_2`、`Linear::bert_cipher_plain_bsgs*`、`Linear::bert_cipher_cipher_cross_packing`、`Linear::softmax_v` 或 BOLT 的 `ss_to_he_*`/`he_to_ss_*`。
4. Softmax 和 LayerNorm 的统计部分确实调用了 BOLT/SCI 的 MPC 函数；GELU 也为 MPC，但调用的是 `gelu_approx`，而 BOLT BERT 与 `BOLT-gelu` 测试使用的是 `gelu_approx_2`。
5. 默认运行未发现中间激活被完整重构为明文；但是 `debug_logits=1` 会把完整 logits 重构给 BOB。烟雾 fixture 和明文参考生成器不是训练 checkpoint，也不能证明 checkpoint 架构或协议语义一致。
6. 输出的 `total_bytes` 排除了 BFV 公钥交换和 base OT setup；双方发送字节相加不是重复计数，但 `rounds_sum` 是双方 round counter 直接相加，不是标准的单边/逻辑轮数。

此外，当前 `BOLT/` 目录没有 `.git` 元数据，无法从该目录独立验证 HEAD、分支名或 commit。因此本文审计的是当前文件快照，不能用 Git 证据确认它确为 `client_opt` 的某个确定提交。

## 2. 从 `run_vit_e2e.sh` 开始的完整调用链

### 2.1 进程和目标

```text
scripts/run_vit_e2e.sh
├─ ALICE: SCI/build_vit/bin/BOLT_VIT r=1 model=alice_model.txt
└─ BOB:   SCI/build_vit/bin/BOLT_VIT r=2 input=bob_image.txt
          └─ diff expected_prediction.txt prediction.txt
```

脚本只校验最终类别文本是否一致。该检查不能证明 packing、BSGS、HE/MPC 转换或 MatMul 后端正确。

`SCI/tests/vit_bolt/CMakeLists.txt` 中：

```cmake
add_executable(BOLT_VIT bolt_vit.cpp vit_model.cpp)
target_link_libraries(BOLT_VIT SCI-FloatingPoint SCI-HE)
```

相比之下，BOLT BERT 通过 `LIB_BERT` 显式编译并链接 `he.cpp`、`nonlinear.cpp`、`linear.cpp`、`bert_utils.cpp` 和 `bert.cpp`。ViT 目标没有链接这些 BOLT Transformer 适配源文件。对已构建二进制执行符号检查也未找到 `Linear::linear_1`、`Linear::linear_2`、`bert_cipher_plain_bsgs*`、`bert_cipher_cipher_cross_packing`、`softmax_v` 或 `w_ln`。

### 2.2 `main` 初始化链

```text
main                                      bolt_vit.cpp:550
├─ load_vit_model                         vit_model.cpp:23      [ALICE]
├─ IOPack                                 io_pack.h:17
│  ├─ NetIO(port)
│  ├─ NetIO(port+50)
│  └─ NetIO(port+100)
├─ OTPack                                 ot_pack.h:43
│  └─ SetupBaseOTs                        ot_pack.h:67
├─ FPMath                                 SCI FloatingPoint/MPC
├─ 发送 WireConfig                         bolt_vit.cpp:572-592
├─ BfvLinear                              bolt_vit.cpp:78
│  ├─ SEAL BFV context
│  ├─ BOB 生成 secret/public key
│  └─ BOB 仅发送 public key
└─ run_vit                                bolt_vit.cpp:455
```

`BfvLinear` 没有创建 `BatchEncoder`、Galois keys 或 relinearization keys。这与后续代码完全没有 batching、rotation、BSGS 和 BFV ciphertext-ciphertext multiplication 相互印证。

### 2.3 `run_vit` 算子链

```text
run_vit
├─ patches                                本地 patch 重排
├─ BfvLinear::private_linear
│  └─ BfvLinear::linear                   patch projection
├─ CLS token + position                   ALICE 的加法分享
├─ 每个 Transformer block
│  ├─ secure_layer_norm                   LN1
│  │  ├─ FPMath::layer_norm_fix           MPC 统计/归一化
│  │  └─ BfvLinear::shared_linear         gamma 对角矩阵，逐标量 BFV
│  ├─ attention
│  │  ├─ BfvLinear::shared_linear ×3      Q/K/V，逐标量 BFV ct-pt
│  │  ├─ secure_matmul                    QKᵀ
│  │  │  └─ LinearOT::matrix_multiplication（通用 OT MatMul）
│  │  ├─ multiply_public                  1/sqrt(head_dim)
│  │  ├─ secure_softmax
│  │  │  └─ FPMath::softmax_fix           BOLT/SCI MPC
│  │  ├─ secure_matmul                    Softmax×V
│  │  │  └─ LinearOT::matrix_multiplication（通用 OT MatMul）
│  │  └─ BfvLinear::shared_linear         输出投影，逐标量 BFV ct-pt
│  ├─ add_secret                          attention residual
│  ├─ secure_layer_norm                   LN2，同 LN1
│  ├─ BfvLinear::shared_linear            FC1，逐标量 BFV ct-pt
│  ├─ FPMath::gelu_approx                 SCI MPC，但非 BOLT gelu_approx_2
│  ├─ BfvLinear::shared_linear            FC2，逐标量 BFV ct-pt
│  └─ add_secret                          FFN residual
├─ secure_layer_norm                      final LN
├─ BfvLinear::shared_linear               分类头，逐标量 BFV ct-pt
├─ secure_argmax                          SCI MPC 比较/选择
├─ reveal_logits                          仅 debug_logits=1
└─ reveal_one                             公开最终类别给 BOB
```

## 3. 逐算子后端审计

下表中的 PASS 只表示该行算子自身进入了预期密码原语，不表示整个 ViT 协议通过。

| operator | expected BOLT backend | actual backend | source file | function name | status | modification required |
|---|---|---|---|---|---|---|
| 运行入口/链接 | ViT 目标应链接适配后的 BOLT `Linear`、`HE`、conversion 与 `NonLinear` 路径 | 只编译 `bolt_vit.cpp`、`vit_model.cpp` 并链接通用 SCI 库；未链接 `bert_bolt/linear.cpp` 等 | `SCI/tests/vit_bolt/CMakeLists.txt:1-9`；对照 `SCI/tests/bert_bolt/CMakeLists.txt:1-16` | `BOLT_VIT` target | **FAIL** | 将 ViT 目标接入经维度泛化的 BOLT Linear/HE/conversion 源码，而不是独立 BFV adapter |
| 图像 patchify | 输入持有方本地执行不含权重的 reshape/permutation；之后才进入安全 projection | BOB 对私有像素本地重排；ALICE 对零向量执行同样重排 | `SCI/tests/vit_bolt/bolt_vit.cpp:374-394,462-465` | `patches` | **PASS** | 无密码后端修改；需明确 patch 顺序与真实 checkpoint 一致 |
| Patch projection | BOLT packed BFV ciphertext-plaintext MatMul；复用/泛化 compact packing、BSGS 和 HE→share | 每个输入标量一个 BFV ciphertext；每个输出逐项 `multiply_plain` 后求和 | `SCI/tests/vit_bolt/bolt_vit.cpp:104-205,465-469` | `BfvLinear::private_linear` → `BfvLinear::linear` | **FAIL** | 为 patch 矩阵建立 FCMetadata/packing，调用泛化后的 `bert_cipher_plain_bsgs_2`/`linear_2` 和 BOLT conversion |
| CLS token/position embedding | 保持 additive-share/packed layout，不公开激活 | ALICE 在自己的 share 上加入 CLS 和 position；未重构 | `SCI/tests/vit_bolt/bolt_vit.cpp:471-484` | `run_vit` | **PASS** | 适配 BOLT packed layout 后需以相同 packing 加入 |
| LayerNorm 1 | BOLT MPC `FPMath::layer_norm_fix` 计算统计；gamma 用 BOLT packed BFV `w_ln`，beta 保持分享 | 统计部分确实调用 `layer_norm_fix`；gamma 作为稠密对角矩阵进入逐标量 `BfvLinear`，beta 加到 ALICE share | `SCI/tests/vit_bolt/bolt_vit.cpp:243-284,490-492`；`SCI/src/FloatingPoint/fp-math.cpp:1708-1772` | `secure_layer_norm`、`FPMath::layer_norm_fix`、`BfvLinear::shared_linear` | **PARTIAL** | 保留 MPC 统计部分；gamma/beta 改接 BOLT `ss_to_he_*` → `Linear::w_ln` → `he_to_ss_*`，并保持 packing/模数转换 |
| Q projection | BOLT packed BFV ct-pt QKV fusion/BSGS | 独立逐标量 BFV ct-pt | `SCI/tests/vit_bolt/bolt_vit.cpp:327-335,135-205` | `attention` → `BfvLinear::shared_linear` | **FAIL** | 复用并泛化 `Linear::linear_1` / `bert_cipher_plain_bsgs`，保持 QKV fused packing |
| K projection | 同 Q | 同 Q | 同上 | 同上 | **FAIL** | 同 Q |
| V projection | 同 Q；V 应保持 BOLT 为后续 ct-ct Softmax×V 准备的 ciphertext packing | 先转成普通 additive `FixArray`，丢失 BOLT V ciphertext/packing | 同上 | 同上 | **FAIL** | 保留 `linear_1` 产生的 packed encrypted V，直到 BOLT `softmax_v` |
| Head split/transpose | 仅重排 secret shares/packed slots，不公开 | 对 `FixArray.data` 的本地 share 做索引复制 | `SCI/tests/vit_bolt/bolt_vit.cpp:336-352` | `attention` | **PASS** | BOLT packed 适配后应改为对应 slot/packing 变换，不能先拆为标量 shares |
| QKᵀ | BOLT BFV ciphertext-ciphertext cross packing：`Linear::bert_cipher_cipher_cross_packing`，含 relinearization/rotations | `secure_matmul` 调用通用 SCI `LinearOT::matrix_multiplication`；底层为 OT cross terms + 本地 share products，不包含 SEAL ct-ct | `SCI/tests/vit_bolt/bolt_vit.cpp:287-298,353-354`；`SCI/src/LinearOT/linear-ot.cpp:577-780` | `secure_matmul` → `LinearOT::matrix_multiplication` | **FAIL** | 使用泛化后的 `Linear::linear_1`/`bert_cipher_cipher_cross_packing` 直接生成 packed QKᵀ ciphertext，再按 BOLT HE→share 转换 |
| Attention scale | 保持 BOLT 层级定点位宽/scale 调度和转换语义 | 在 37-bit、scale=12 的 generic `FixArray` 上乘公开定点常数并 truncate | `SCI/tests/vit_bolt/bolt_vit.cpp:300-306,355-356` | `multiply_public` | **PARTIAL** | 明确 ViT 的 `1/sqrt(d)` 在 BOLT scale schedule 中的位置；使用与 QK 输出模数、`gt_p_sub` 和 right shift 一致的调度 |
| Softmax | BOLT MPC：`NonLinear::softmax` → `FPMath::softmax_fix` | 直接调用同一个 `FPMath::softmax_fix`；密码函数匹配，但只使用单个 FPMath 实例 | `SCI/tests/vit_bolt/bolt_vit.cpp:308-325,357`；`SCI/src/FloatingPoint/fp-math.cpp:1223-1307` | `secure_softmax` → `FPMath::softmax_fix` | **PASS** | 原语无需替换；恢复 BOLT 前后 packing/conversion，性能复现时恢复同等并行切分 |
| Softmax×V | BOLT BFV ciphertext-ciphertext：`preprocess_softmax_s1_ct_ct` → `Linear::softmax_v` → `Evaluator::multiply` | `secure_matmul` 再次调用通用 SCI/OT MatMul；Softmax 和 V 已是普通 additive shares | `SCI/tests/vit_bolt/bolt_vit.cpp:358-359`；`SCI/src/LinearOT/linear-ot.cpp:577-780` | `secure_matmul` → `LinearOT::matrix_multiplication` | **FAIL** | 按 BOLT diagonal/cross packing 将 Softmax share 转回 ciphertext，并调用 `Linear::softmax_v` 保持 ct-ct 路径 |
| Multi-head merge | 保持秘密表示的 layout 变换 | 本地复制 additive shares，未重构 | `SCI/tests/vit_bolt/bolt_vit.cpp:360-365` | `attention` | **PASS** | 改为适配后的 BOLT packed postprocess/layout，不应成为 unpack-to-scalar 路径 |
| Attention 输出投影 | BOLT packed BFV ct-pt BSGS：`Linear::linear_2` → `bert_cipher_plain_bsgs_2` | 逐标量 `BfvLinear::shared_linear` | `SCI/tests/vit_bolt/bolt_vit.cpp:366-371` | `BfvLinear::shared_linear` | **FAIL** | 泛化并调用 BOLT `linear_2`，保留 compact packing、BSGS、flood/mod-switch 与 conversion |
| Attention residual | additive-share addition，保持 scale/ell | `FixOp::add`；不公开 | `SCI/tests/vit_bolt/bolt_vit.cpp:216-220,493-494` | `add_secret` | **PASS** | 接入 packed linear 后校验 residual 两支的 layout、ell、scale 一致 |
| LayerNorm 2 | 同 LayerNorm 1 | 同 LayerNorm 1 | `SCI/tests/vit_bolt/bolt_vit.cpp:495-497` | `secure_layer_norm` | **PARTIAL** | 同 LayerNorm 1 |
| FC1 | BOLT packed BFV ct-pt BSGS：`Linear::linear_2` | 逐标量 `BfvLinear::shared_linear` | `SCI/tests/vit_bolt/bolt_vit.cpp:500-504` | `BfvLinear::shared_linear` | **FAIL** | 泛化 BOLT FC metadata/BSGS，使用 `linear_2` 和原 conversion/scale schedule |
| GELU | BOLT MPC BERT 路径：`NonLinear::gelu` → `FPMath::gelu_approx_2`，并使用 BOLT GELU 位宽/scale | 调用 MPC `FPMath::gelu_approx`；仍是安全计算，但多项式/分段函数与 BOLT BERT 实际路径不同，且统一使用 ell=37、scale=12 | `SCI/tests/vit_bolt/bolt_vit.cpp:505-510`；对照 `SCI/tests/bert_bolt/nonlinear.cpp:181-219`、`SCI/src/FloatingPoint/fp-math.cpp:1437-1508,1510+` | `FPMath::gelu_approx` | **FAIL** | 调用 BOLT 实际 `gelu_approx_2`，恢复 `GELU_ELL=20`、`GELU_SCALE=11` 及前后 reduce/shift 语义，明文参考使用相同近似 |
| FC2 | BOLT packed BFV ct-pt BSGS：`Linear::linear_2` | 逐标量 `BfvLinear::shared_linear` | `SCI/tests/vit_bolt/bolt_vit.cpp:511-515` | `BfvLinear::shared_linear` | **FAIL** | 同 FC1，并校验输出与 residual 的 scale/layout |
| FFN residual | additive-share addition，保持 scale/ell | `FixOp::add`；不公开 | `SCI/tests/vit_bolt/bolt_vit.cpp:516` | `add_secret` | **PASS** | 接回 BOLT conversion 后校验 metadata |
| Final LayerNorm | 若 checkpoint 为 pre-norm ViT，应使用与前两处相同的 BOLT LN MPC + packed gamma | 统计为 BOLT MPC，gamma 为逐标量 BFV adapter | `SCI/tests/vit_bolt/bolt_vit.cpp:519-521` | `secure_layer_norm` | **PARTIAL** | 保留 final norm 但改用 BOLT LN affine/conversion；从 checkpoint 固化 epsilon |
| 分类头 | 需先固定适配规范：若视为 ViT linear，应使用 BOLT packed BFV ct-pt；BOLT BERT 原 pool/class 则使用 MPC `p_matrix_mul_iron` | 逐标量 BFV adapter；既不是 BOLT packed ct-pt，也不是 BOLT BERT pool/class 路径 | `SCI/tests/vit_bolt/bolt_vit.cpp:522-528`；对照 `SCI/tests/bert_bolt/bert.cpp:1760-1847` | `BfvLinear::shared_linear` | **FAIL** | 明确比较口径后复用相应 BOLT 后端，不能保留独立 scalar adapter |
| Argmax/最终公开 | 与所选 BOLT 输出语义一致；只在协议规定边界公开 | SCI `GT`/`if_else` 安全 argmax 后只重构类别。BOLT BERT 原实现会重构 logits 给 BOB，当前行为更强隐私但改变协议与通信口径 | `SCI/tests/vit_bolt/bolt_vit.cpp:396-422,529-536`；对照 `SCI/tests/bert_bolt/bert.cpp:1923-1934` | `secure_argmax`、`reveal_one` | **PARTIAL** | 在复现实验规范中固定输出边界；若要求严格可比，应采用同一 disclosure 和计量范围 |
| Debug logits | 不得在正式协议中重构中间/最终向量，除非明确属于输出 | `debug_logits=1` 时 ALICE 发送 logits share，BOB 重构完整 logits | `SCI/tests/vit_bolt/bolt_vit.cpp:424-447,530,562` | `reveal_logits` | **FAIL**（条件性） | 正式复现禁用并在运行时/构建时强约束；调试结果不得计入安全或性能结论 |
| HE↔MPC conversion | BOLT `ss_to_he_server/client`、`he_to_ss_server/client`，保持 batch packing、随机 mask、plain-prime↔2^ell 纠正 | 自定义 `BfvLinear::linear`：逐标量 encrypt/add-share、输出随机 mask、BOB decrypt；随后直接 `truncate_reduce` | `SCI/tests/vit_bolt/bolt_vit.cpp:135-205`；对照 `SCI/tests/bert_bolt/bert.cpp:201-315` | `BfvLinear::linear` | **FAIL** | 删除独立 conversion，复用 BOLT packed conversion 与 `gt_p_sub`/right-shift 调度 |
| 固定点/明文模数 | 保留 BOLT 各阶段 BFV prime、37/12 主语义、GELU 20/11、QK/linear 的显式 rescale 与 p→ring 转换 | 所有层统一 BFV plaintext modulus `2^37`、ell=37、scale=12；没有 BOLT `gt_p_sub`，没有原分层 scale schedule | `SCI/tests/vit_bolt/bolt_vit.cpp:36,50-53,81-85,200-204`；对照 `SCI/tests/bert_bolt/linear.cpp:30-63`、`bert.h:11-16`、`bert.cpp:800-840,1348-1398` | `encode_ring`、`BfvLinear`、`truncate_reduce` | **FAIL** | 按 BOLT 原参数和每层量化计划做 ViT 维度适配；不可用统一环模数规避 BOLT conversion |

## 4. BOLT 参考后端的实际密码函数

本审计不是依据函数或日志名称判定，而是继续追踪到 SEAL/SCI 操作：

### 4.1 BOLT ciphertext-plaintext MatMul

- `Linear::linear_1` → `Linear::bert_cipher_plain_bsgs`
  - `BatchEncoder` 编码 compact/cross-packed 权重；
  - `Evaluator::rotate_rows` / `rotate_columns`；
  - `Evaluator::multiply_plain`；
  - 按 BSGS 的 baby/giant step 累加。
- `Linear::linear_2` → `Linear::bert_cipher_plain_bsgs_2`
  - 同样使用 packing、rotation、`multiply_plain`、BSGS；
  - 输出执行 flooding 和 modulus switching。

对应源码：`SCI/tests/bert_bolt/linear.cpp:316-363,730-997`。

当前 ViT 的 `BfvLinear::linear` 只有逐标量 `Encryptor::encrypt`、`Evaluator::multiply_plain` 和 `Evaluator::add`，没有上述 packing/rotation/BSGS。

### 4.2 BOLT ciphertext-ciphertext MatMul

- QKᵀ：`Linear::bert_cipher_cipher_cross_packing`
  - `Evaluator::multiply(Qi, rotated(Ki))`；
  - `relinearize_inplace`；
  - rotation、cross mask 和累加。
- Softmax×V：`Linear::softmax_v`
  - packed Softmax ciphertext 与 packed V ciphertext 进入 `Evaluator::multiply`；
  - relinearization 和 modulus switching。

对应源码：`SCI/tests/bert_bolt/linear.cpp:1004-1062,1734-1775,1803-1868`。

当前 ViT 的两处 `secure_matmul` 都进入 `LinearOT::matrix_multiplication`。该函数通过 OT 计算双方 share 的 cross terms，同时用 `matmul_cleartext` 计算各方**本地 share 的局部乘积**；这里的 `matmul_cleartext` 不是完整明文重构，但整个后端是 generic MPC MatMul，而不是 BOLT BFV ct-ct。

## 5. generic、fallback、mock、stub 与明文重构检查

### 5.1 发现的位置

| 类型 | 位置 | 判定 |
|---|---|---|
| 通用 MatMul | `bolt_vit.cpp:287-298` 的 `secure_matmul` | QKᵀ 和 Softmax×V 均回退到 `LinearOT::matrix_multiplication`；违反协议保持要求 |
| 统一 Attention 函数 | `bolt_vit.cpp:327-372` 的 `attention` | 没有复用外部 cleartext Attention 类，但该统一函数将 QKV、QK、Softmax×V、输出投影组合为 scalar-share 流程；其中两个 MatMul 是 generic MPC |
| 自定义线性 fallback | `bolt_vit.cpp:78-214` 的 `BfvLinear` | 不是明文线性，但它绕过 BOLT packed/BSGS Linear，属于协议后端 fallback |
| 明文测试参考 | `scripts/generate_vit_fixture.py:24-138` | 离线 cleartext reference，仅生成随机烟雾 fixture；不在两方协议调用链中 |
| mock/stub | 运行路径未发现名为 mock/stub 的分支 | 没有运行时 mock 算子；但随机 tiny fixture 只是测试桩性质的数据，不能替代 checkpoint 复现 |
| debug 重构 | `bolt_vit.cpp:424-447` | `debug_logits=1` 时重构完整 logits 给 BOB |

### 5.2 默认路径是否重构中间值

默认 `DEBUG_LOGITS=0` 时，未发现 Q、K、V、QK、Softmax、attention output、LayerNorm output、FC1 或 FC2 被完整重构：

- `BfvLinear::linear` 中，ALICE 对 HE 输出加入随机 mask并持有 `-mask` share；BOB 只解密 masked output并持有另一 share。只要 mask 未泄漏，BOB 得不到完整中间值。
- `FixArray.data` 保存的是两方 additive shares，本地索引复制、bias/residual 加法不构成重构。
- `LinearOT::matmul_cleartext` 只乘本地 shares，完整乘积仍由 local terms + OT cross terms 组成，不是 plain fallback。
- `reveal_one` 只重构最终类别，是默认输出边界。

另外，`BfvLinear::plain`（`bolt_vit.cpp:129-133`）把权重、mask 或输入 share 编码为 SEAL plaintext，`diagonal`（234-241）在 ALICE 侧准备私有 gamma 对角权重，`add_bias`（222-232）只修改 ALICE 的输出 share。这些位置本身没有重构完整激活，不能标成 cleartext inference；问题在于它们共同构成了绕过 BOLT packing/BSGS 的 scalar HE fallback。

需要注意的显式公开：

- `debug_logits=1`：向 BOB 重构完整 logits；
- ALICE 向 BOB 发送模型尺寸 `WireConfig`；
- run script 用明文期望类别做协议外 `diff`，只验证测试结果。

## 6. compact packing、BSGS、转换和固定点语义

### 6.1 compact packing / BSGS

当前 ViT 路径为 **FAIL**：

- `BfvLinear` 没有 `BatchEncoder`；
- 一个输入标量对应一个 BFV ciphertext；
- 没有 Galois keys、slot rotations 或 relinearization keys；
- 没有调用 `bert_cross_packing_*`；
- 没有调用 `bert_cipher_plain_bsgs*`；
- 没有调用 `bert_cipher_cipher_cross_packing` 或 `softmax_v`。

因此高通信量不是 BOLT ViT 的性能结果，而是 scalar BFV adapter 的结果。

### 6.2 HE/MPC conversion

当前转换具有随机 masking，默认不直接泄漏完整激活，但不是 BOLT conversion：

- BOLT 使用 BatchEncoder-packed `ss_to_he_*` / `he_to_ss_*`；
- BOLT BFV plaintext modulus 是按阶段选择的 prime，并在进入 2^ell sharing 后执行 `gt_p_sub`、right shift/reduce；
- 当前实现将 BFV plaintext modulus直接设为 `2^37`，然后把解密 residue直接包装为 37-bit `FixArray` 并 truncation。

这改变了模数域、packing、conversion 通信和数值误差路径，不能用于协议保持对比。

### 6.3 固定点

当前主张 `ell=37, scale=12`，但没有保留 BOLT 的逐层语义：

- BOLT BERT 的 GELU 输入使用 `GELU_ELL=20, GELU_SCALE=11`；当前 GELU 为 37/12。
- BOLT QK 输出根据层使用显式 21/22→12 的 shift；当前对所有层使用一次公开常数乘法和统一 truncation。
- BOLT linear/LN 使用不同 BFV plaintext primes；当前只有 `2^37`。
- 明文 generator 使用 `math.erf` 的精确 GELU，而密态使用 `gelu_approx`，所以 reference logits 不是同一近似图。

## 7. 通信量与轮数口径

### 7.1 `total_bytes`

`before_bytes` 在 `run_vit` 入口才读取，而以下动作都发生在它之前：

1. `IOPack` 三个连接建立；
2. `OTPack` 构造和 `SetupBaseOTs`；
3. `FPMath` 构造；
4. `WireConfig` 发送；
5. `BfvLinear` 的 BFV context、key generation 和 public-key 发送。

因此 `E2E_RESULT total_bytes`：

- **不包含** base OT setup；
- **不包含** BFV public-key exchange；
- **不包含**配置元数据；
- 包含 `run_vit` 内的 scalar HE ciphertext、在线 OT/MPC、最终 reveal；若开启 debug，也包含 logits reveal；
- 不包含结尾双方交换 `RunStats` 的字节。

`NetIO::counter` 只在 `send_data` 增加，`recv_data` 不增加。最终 `local.bytes + peer.bytes` 是两方发送字节的合计，即双向总线上行字节，**不是把同一发送和接收重复统计两次**。

### 7.2 `rounds_sum`

`NetIO::num_rounds` 在 send/recv 方向切换时都会增加；每方都有自己的 counter。当前代码把 ALICE 与 BOB 的 local rounds直接相加。因此 `rounds_sum` 通常近似把一个逻辑交互从双方各计一次，不能直接当作 BOLT 报告中的协议轮数。`IOPack::get_rounds` 还按实现约定忽略 `io_rev` 的 rounds，因为其被视为与主 channel 并行。

为了可比，应报告：

- setup 与 online 分开的通信；
- 是否含 BFV keys、base OT、model metadata；
- 单边 send bytes 或双向总 send bytes的统一定义；
- 单边/关键路径 logical rounds，而不是双方 counter 求和。

## 8. Pre-norm / Post-norm 与 checkpoint 一致性

当前 ViT 明确是 **pre-norm**：

```text
x = x + Attention(LN1(x))
x = x + FC2(GELU(FC1(LN2(x))))
logits = Head(FinalLN(x)[CLS])
```

证据为 `run_vit` 中 LN1 位于 attention 前、LN2 位于 FFN 前，并存在 final LN：`bolt_vit.cpp:486-526`。`generate_vit_fixture.py:100-136` 使用相同 pre-norm 顺序，因此当前随机 fixture 与当前 C++ 拓扑一致。

BOLT BERT 原图是 post-norm 风格：attention/FFN 输出先与 residual 相加，再进入相应 LayerNorm。ViT 不需要机械复制 BERT 的 norm 位置，但必须服从目标 checkpoint。

**当前无法确认 checkpoint 一致性，状态为 FAIL/未验证：**

- 仓库内没有 `.pt`、`.pth`、`.onnx`、`.safetensors` 或其他训练 ViT checkpoint；
- `alice_model.txt` 来自固定随机数生成器，不是训练模型；
- 模型格式没有 LayerNorm epsilon 字段，`layer_norm_fix` 也没有显式加 checkpoint epsilon；
- `VitBlockModel` 没有 Q/K/V bias 字段，许多常见 ViT checkpoint 的 QKV projection 含 bias；
- 没有 checkpoint 名称、patch convention、输入归一化、类别映射等架构元数据；
- 明文参考用 exact erf GELU，而密态图用多项式近似。

所以只能说“当前 C++ 与自生成 fixture 都是 pre-norm”，不能说“与目标 checkpoint 架构一致”。

## 9. 必须完成的协议保持修改清单

按阻断程度排序：

1. 删除/停用 ViT 路径中的 scalar `BfvLinear`，让 ViT 目标链接并调用泛化后的 BOLT `Linear`/`HE`/conversion。
2. 为 ViT token 数、hidden size、head size、FFN size 和 patch matrix 泛化 `FCMetadata`、compact/cross packing 与 BSGS 参数；不能退化为逐标量 ciphertext。
3. Q/K/V 使用 `linear_1` 等价的 fused packed ct-pt；QKᵀ必须进入 `bert_cipher_cipher_cross_packing` 等价路径。
4. Softmax 输出按 BOLT conversion/diagonal packing 转回 HE，Softmax×V必须进入 `Linear::softmax_v` 等价 ct-ct 路径。
5. 输出投影、FC1、FC2、patch projection 和所选分类头使用 BOLT packed ct-pt `linear_2` 等价路径。
6. LayerNorm 保留 `layer_norm_fix`，gamma 改用 BOLT packed `w_ln`；恢复 prime↔ring conversion、bias 和 scale 调度。
7. GELU 改为 BOLT 实际 `gelu_approx_2` 与对应 20/11 定点语义；明文 reference 使用同一近似用于协议数值校验。
8. 引入真实 checkpoint 导出器并固定 pre/post-norm、LN epsilon、QKV bias、patch 顺序、输入归一化和类别映射。
9. 重做统计边界：分别给出 setup、online，统一字节和 rounds 口径；默认禁止 debug logits。

在第 1—7 项完成前，即使 `run_vit_e2e.sh` 输出 `PASS`，也只能说明随机 tiny fixture 的最终类别一致，不能作为 BOLT 协议保持或 BOLT-ViT 性能复现的证据。
