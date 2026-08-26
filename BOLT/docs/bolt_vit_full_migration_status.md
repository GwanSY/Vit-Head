# BOLT-ViT 完整迁移状态与阻断项

日期：2026-08-04。范围仅限本 `BOLT/` 目录。

## 已完成并实测

- 正式 `BOLT_VIT` 已接入完整 pre-norm 多层图：Stem、LN1、Attention、
  residual、LN2、FC1、`gelu_approx_2`、FC2、residual、Final LN、CLS、
  packed classifier 和 MPC argmax。
- Patch/QKV/output/FFN/classifier 使用共享 BFV ct-pt compact BSGS；QK 和
  Softmax×V 使用共享 BFV ct-ct 内核。Q/K/V 在 MPC truncation 后立即回到
  packed BFV，V ciphertext 保留到 Softmax×V。
- LN2/Final LN 的 statistics 使用 `FPMath::layer_norm_fix`，gamma 使用
  packed diagonal ct-pt affine，beta 为 ALICE 本地 share addition。
- GELU 固定执行 ell37/scale12 → ell20/scale11 → BOLT
  `gelu_approx_2` → ell37/scale4；FC2 恢复到 scale12。
- 两个 residual 的 trace 都是 0 bytes/0 rounds。
- 单层 tiny 最终类 1 与 fixture 一致；两层 tiny 最终类 0 与 fixture 一致。
- 10 个固定 seed 的完整 secure final-class 收集完成：9/10 与 FP32
  fixture 一致，seed 20260814 不一致，因此总体明确为 FAIL，而非调大容差。
- M1 BERT 三个 hash 与操作计数未变；M2、M3-A 和三内核回归保持 PASS。
- 200-key DeiT checkpoint schema 转换和 C++ strict loader 通过；没有 fuzzy
  matching，原始 checkpoint SHA256 固定。
- 性能统计拆分 context/secret-key generation、evaluation/public keys、base
  OT、public metadata、offline、online、final reveal。bytes 是两端 send-only
  counter 之和，logical rounds 是端点最大值而非相加。
- generic Transformer OT MatMul 现在有实际运行期 counter；默认完整运行为
  0，非零立即抛异常。

## 关键实现位置

| 部分 | 文件 | 函数 |
|---|---|---|
| 正式两方入口、ownership、phase accounting、final reveal | `SCI/tests/vit_bolt/bolt_vit.cpp` | `main` |
| 完整图调度 | `SCI/tests/vit_bolt/bolt_vit_protocol.cpp` | `BoltVitProtocol::run` |
| packed ct-pt 与 HE/MPC truncation | 同上 | `packed_linear` |
| 三处 LayerNorm | 同上 | `layer_norm` |
| QKV/QK/Softmax/Softmax×V/output | 同上 | `attention` |
| BOLT GELU contract | 同上 | `gelu_bolt` |
| residual | 同上 | `residual_add` |
| BFV setup/accounting | `SCI/src/BOLTTransformer/bolt_he.cpp` | `BoltHE::BoltHE` |
| runtime generic OT audit | `SCI/src/LinearOT/linear-ot.cpp` | `LinearOT::matrix_multiplication` |
| strict checkpoint conversion | `scripts/convert_deit_checkpoint.py` | `main` |
| independent FP32 graph | `scripts/evaluate_deit_fp32.py` | `DeiTNumpy` |

## 当前不能给最终 PASS 的阻断项

1. `SCI/tests/vit_bolt/bolt_vit_protocol.cpp` 尚未提供 validation-only observer，
   因而还不能输出完整图所有中间 share 给独立验证 target。当前独立 B/C
   numerical coverage 到 M3-A Stem/LN1 和 M2 完整 Attention；LN2、Final LN、
   FFN、两个完整 residual 和 classifier 缺逐层 B/C CSV。
2. `outputs/vit_seed_regression.csv:6` 已出现 seed 20260814 FP32 class 0、secure
   class 1。由于缺完整 fixed-clear B，尚不能判断差异来自 A↔B 定点近似还是
   B↔C 密码路径，故 `SECURE_FIXED_LAYERWISE_MATCH` 不能设为 PASS。
3. 本目录没有 ImageNet-1k validation 图片和真实标签。`outputs/accuracy/
   accuracy_report.csv` 明确记录三条 accuracy 未完成状态；任意数字都会是伪造。
4. 真实 DeiT 的 schema/loader 已通过，但 12 层、197 token、dim192 的 secure
   运行尚未执行；因此不能把 checkpoint schema conversion 等同于最终
   `CHECKPOINT_INTEGRATION=PASS`。

## 当前最终状态

```text
BOLT_BERT_REGRESSION=PASS
BOLT_VIT_SINGLE_LAYER=NOT_VERIFIED_LAYERWISE
BOLT_VIT_MULTI_LAYER=NOT_VERIFIED_LAYERWISE
CHECKPOINT_INTEGRATION=SCHEMA_AND_LOADER_ONLY
SECURE_FIXED_LAYERWISE_MATCH=NOT_IMPLEMENTED_FOR_FULL_GRAPH
FP32_FIXED_ACCURACY_REPORTED=false
PERFORMANCE_ACCOUNTING_VALID=true
PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0
SCALAR_FALLBACKS=0
GENERIC_OT_MATMUL_CALLS=0
SILENT_FALLBACKS=0
BOLT_VIT_PROTOCOL_PRESERVING=NOT_YET_VERIFIED
```

因此本实现不会打印用户定义的最终总 PASS。
