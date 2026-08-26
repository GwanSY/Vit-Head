# BLB ViT-B/16：实现审计、复现命令与验收状态

本目录为 BLB 新增标准 `google/vit-base-patch16-224` 的端到端适配层。执行图是
ViT 的 pre-LayerNorm 拓扑：patch projection → CLS/position embedding → 12 个
`LN → Attention → residual → LN → MLP → residual` block → final LN → CLS →
768→1000 classifier。它没有把 BERT 的 Add&Norm 顺序照搬到 ViT，也没有新增
HE 卷积。

除非另行说明，以下命令均从
`/home/dm/mind/MIND_code/Pencil/BLB` 执行。结果记录日期为 2026-08-10。

## 1. 仓库审计

### 1.1 BERT 专用硬编码

原 BOLT/BERT 路径位于 `SCI/tests/bert_bolt`，审计到的模型专用约束包括：

- `linear.h` 和 `linear_ckks.h` 以宏固定 `INPUT_DIM=128`、
  `COMMON_DIM=768`、`OUTPUT_DIM=64`、`INTER_DIM=3072`、
  `PACKING_NUM=12` 和 `ATTENTION_LAYERS=1`；
- `bert.h` 固定 `NUM_CLASS=2`，分类路径按 BERT pooler 和二分类头分配；
- `bert.cpp` 直接以 128 读取 attention mask，并按 `128×768`、
  `12×128×128`、`12×128×64` 分配和转置；
- `linear.cpp` / `linear_ckks.cpp` 的旧 cross-packing 含 `% 128`、`j < 32`、
  `j == 32` 等 BERT-128 的布局分支；
- checkpoint reader 按 BERT 的 layer、pooler、classifier 文件布局取权重；
- BERT 的 Add&Norm 拓扑和 ViT checkpoint 的 pre-LN 拓扑不同；
- BERT 原 `softmax_fix` 的窄位宽求和不适合一行 197 项。ViT 适配层直接调用已有
  `FPMath::softmax_fix_iron_1` 的 37-bit 路径，没有修改 Softmax 的协议定义。

这些值没有被写回密码学后端成为新的全局常量；ViT loader 和执行图从模型
metadata 取得 runtime shape。标准快速路径对真实 B/16 shape 作 fail-closed 检查。

### 1.2 直接复用的 BOLT 入口

ViT 执行图复用的既有实现及其薄适配入口如下：

| ViT 运算 | 复用入口 |
|---|---|
| SS↔BFV | BOLT HE↔MPC conversion；`BoltVitCompatConversion::ss_to_he_arbitrary_slots_signed` 仅适配任意 ciphertext 数和 signed `Z2^37` slots |
| patch/Q/K/V/out/MLP ct-pt MatMul | BERT BSGS，经 `BoltLinear::bert_cipher_plain_matmul_bsgs_compat` / `bert_qkv_cipher_plain_matmul_bsgs_compat` 调用 |
| QKᵀ ct-ct MatMul | BERT cross packing，经 `BoltLinear::bert_qk_matmul_cross_compat` 调用 |
| Softmax×V ct-ct MatMul | BERT softmax/V packing，经 `BoltLinear::bert_softmax_v_matmul_compat` 调用 |
| Softmax | `FPMath::softmax_fix_iron_1` |
| GELU | `FPMath::gelu_approx_2`，输入 ell=20/scale=11，输出 ell=37/scale=4 |
| LayerNorm | `FPMath::layer_norm_fix`，gamma/beta 的 affine 仍走 ct-pt 路径 |
| truncation / ArgMax / public mask | 既有 `FixOp` / MPC / ct-pt 与公开二值 mask 操作 |

CLS 与 position embedding 是 ALICE 的模型参数，只以 secret-share 本地加法加入；
BOB 不读取模型文件。BOB 本地 patchify 只是固定公开重排，768→768 projection 仍由
密文-明文 MatMul 执行。

### 1.3 新增文件分组

- 模型与 checkpoint：`vit_model.{h,cpp}`、`vit_model_inspect.cpp`、
  `scripts/convert_vit_checkpoint.py`、`scripts/convert_deit_checkpoint.py`、
  `scripts/verify_vit_manifest.py` 和 `scripts/test_vit_checkpoint_loader.py`；
- 执行图：`blb_vit_protocol.{h,cpp}`、`blb_vit.cpp`；
- validation-only 聚合误差：`vit_validation_observer.{h,cpp}`、
  `scripts/prepare_vit_oracle_shares.py`、`scripts/test_vit_oracle_shares.py`、
  `scripts/validate_vit_aggregate_errors.py`、
  `scripts/test_vit_metric_invariants.py` 和
  `scripts/export_vit_layer_errors.py`；
- BOLT 薄适配层：`BOLTTransformer/bolt_{metadata,he,conversion,linear,attention}.*`
  和 `BOLTTransformer/bolt_vit_compat_{packing,conversion,linear,attention}.*`；
- clear/fixed/reference：`scripts/preprocess_vit_image.py`、
  `scripts/vit_pytorch_reference.py`、`scripts/vit_reference.py`、
  `scripts/audit_bfv_ranges.py`、`scripts/compare_vit_logits.py`、
  `scripts/evaluate_vit_b16_pytorch.py` 和
  `scripts/summarize_vit_acceptance.py`；
- 测试：`BOLTTransformer/*_test.cpp`、`bolt_signed_conversion_test.cpp`、
  `scripts/test_vit_reference_nonlinear.py`、`scripts/test_audit_bfv_ranges.py`；
- 复现入口：`repro/build_vit_b16.sh`、`repro/run_vit_b16.sh`、
  `repro/run_vit_fixture.sh` 和 `repro/run_vit_seed_regression.sh`。

没有改写 BFV、MPC、OT、HE↔MPC、Softmax、GELU、LayerNorm、MatMul、BSGS、
key switching 或 packing 的协议数学定义。

### 1.4 两处 SCI 源码差异

本轮 ViT 实施对 SCI 核心有两处需要披露的源码差异；工作树中的其他既有改动不在
本文归因范围内：

1. `SCI/src/LinearOT/linear-ot.cpp` 恢复
   `#pragma omp parallel num_threads(2)`。该并行区本来负责 secret-secret 乘法的两个
   cross term；注释它会只执行其中一项。恢复并行执行不改变公式或安全模型。
2. `SCI/src/FloatingPoint/fp-math.cpp` 删除 `gt_p_sub` 中两条无条件打印
   `gt.data[0]` / `sub.data[0]` secret share 的调试输出。计算不变，避免日志泄露。

## 2. 模型、checkpoint 与 loader

真实 checkpoint 为 `google/vit-base-patch16-224`，revision
`10064cc17b69d623a025c9e90e11230cc8ba6967`：

| 文件 | 路径 | SHA256 |
|---|---|---|
| raw safetensors | `../BOLT/checkpoints/vit_b16_hf/model.safetensors` | `1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d` |
| canonical BOLT V3 | `../BOLT/checkpoints/vit_b16/bolt_model.txt` | `c980c11a8478f3ef172fa1121fc87eb02141ad8d8934953ed0323c60fe7d4715` |
| strict manifest | `../BOLT/checkpoints/vit_b16/manifest.json` | 记录 source/revision/raw SHA、真实 shape 和 strict tensor consumption |

loader 覆盖 fused QKV、独立 Q/K/V 和 torchvision 常见命名；权重切分方向、bias、
unexpected/missing tensor 都有负测试。它还 fail-closed 拒绝安全图未实现的
`gelu_new`/`gelu_fast`，并要求 LayerNorm epsilon 在 checkpoint 的 scale=12 下量化
为 0；当前真实 checkpoint 是标准 `gelu`、epsilon=`1e-12`。synthetic fixture 会被标成
`random_weights=true`、`integration_eligible=false`，不能作为真实 integration
证据。

```bash
python3 vit_b16/scripts/test_vit_checkpoint_loader.py
build_vit_b16/adapter/bin/blb_vit_model_inspect \
  ../BOLT/checkpoints/vit_b16/bolt_model.txt
python3 vit_b16/scripts/verify_vit_manifest.py \
  --model ../BOLT/checkpoints/vit_b16/bolt_model.txt \
  --manifest ../BOLT/checkpoints/vit_b16/manifest.json
```

当前 loader 输出 224/16/3、197 tokens、D=768、12 heads、FFN=3072、12 blocks、
1000 classes、`norm_order=pre`、`qkv_bias=1`，状态 PASS。

## 3. 构建与完整推理命令

一条可复现构建命令：

```bash
bash repro/build_vit_b16.sh
```

它构建 production、validation、model inspector 以及 packing、conversion、linear、
encrypted attention、197-way Softmax 和 signed conversion 测试 target。

ViT 保留 CPU/SEAL 默认后端，同时提供独立的 Phantom/CUDA 构建入口。CPU 构建不
链接 Phantom 实现；GPU 构建只让 ALICE（HE evaluator）在 Phantom 上执行 BFV
add、ct-pt/ct-ct multiply、relinearization、mod-switch 和 Galois rotation，BOB
仍负责 host 侧 key generation/encryption/decryption。当前 GPU 适配层保留上层
SEAL ciphertext/wire format，并在每个算子边界 host↔device 搬运，目的是先保证
语义兼容和可回归；后续若追求吞吐，应把中间 ciphertext 改为 device-resident。

```bash
# CPU/SEAL（默认，原入口不变）
JOBS=2 bash repro/build_vit_b16.sh

# GPU/Phantom（使用独立 build_vit_b16_gpu 目录，不覆盖 CPU build）
CUDA_ARCH=70 JOBS=2 bash repro/build_vit_b16_gpu.sh
```

也可以直接调用通用入口并指定 `USE_HE_GPU=ON BLB_VIT_USE_GPU=ON`。GPU 运行机器
需要匹配的 NVIDIA driver；构建只需要 `nvcc` 和 CUDA headers。

完整 12 层 production 推理只揭示最终 class index：

```bash
MODEL=../BOLT/checkpoints/vit_b16/bolt_model.txt \
MANIFEST=../BOLT/checkpoints/vit_b16/manifest.json \
IMAGE=../BOLT/outputs/vit_b16_smoke/parrots.png \
OUTPUT=outputs/vit_b16_secure_real_full12 \
PYTHON=../BOLT/.venv-vit-b16/bin/python \
PORT=49000 RUN_TIMEOUT=0 \
  bash repro/run_vit_b16.sh
```

验收时可额外设 `VALIDATION_LOGITS=1` 使用同一 secure 图并揭示最终 1000 logits；
该 binary 仍不恢复中间 tensor。`BLOCK_LIMIT=1` 只允许与
`VALIDATION_LOGITS=1` 同时用于单 block 验收，production 会拒绝 block limit。

需要复现 27 个逐层聚合误差标量时，显式启用 validation observer（会增加时间和
通信，不能作为 production latency）：

```bash
MODEL=../BOLT/checkpoints/vit_b16/bolt_model.txt \
MANIFEST=../BOLT/checkpoints/vit_b16/manifest.json \
IMAGE=../BOLT/outputs/vit_b16_smoke/parrots.png \
OUTPUT=outputs/vit_b16_secure_real_full12_aggregate_exact \
PYTHON=../BOLT/.venv-vit-b16/bin/python \
VALIDATION_LOGITS=1 AGGREGATE_ERRORS=1 BLOCK_LIMIT=0 \
PORT=53511 RUN_TIMEOUT=8h \
  bash repro/run_vit_b16.sh
```

## 4. shape、scale、bit length 与 packing

主执行图的 arithmetic share 使用 ell=37；主 activation scale=12。validation-only
误差 observer 另用 ell=43/scale=18。完整逐算子记录位于
`operator_trace.csv`，以下是标准图的摘要：

| 阶段 | shape | scale | packing |
|---|---:|---:|---|
| client patchify | `[196,768]` | 12 | `py,px,c,ky,kx` 展平 |
| patch projection / hidden | `[256,768]` | 12 | compact feature-major；token 0..196 valid，197..255 padding |
| LN output | `[256,768]` | 12 | `Z2^37` token-major shares |
| Q / K | `[12,256,64]` | 10 | BERT paired-head compact slots |
| V | `[12,256,64]` | 12 | BERT paired-head compact slots |
| QKᵀ raw | `[12,256,256]` | 20 | BERT cross-diagonal slots |
| Softmax logical submatrix | `[12,197,197]` | 12 | 只归一化 valid query/key；扩回物理布局时 padding 为 0 |
| Softmax×V / attention context | `[256,768]` | 12 | paired-head context → token-major |
| FC1 | `[256,3072]` | 12 | compact split-feature halves |
| GELU | `[256,3072]` | 4 | `Z2^37` shares |
| FC2 / block output | `[256,768]` | 12 | compact → token-major，边界重新施加公开 mask |
| final CLS | `[768]` | 12 | token 0 的 share |
| logits | `[1000]` | 12 | packed classifier output → shares |

patch/Q/K/V/out/fc1 的乘法 raw scale 依其权重分别为 22、23 或 24；fc2 raw
scale=16，LN affine 和 classifier raw scale=23，随后均用已有 MPC truncation 回到
表中 scale。物理 stride 为 patch/attention/classifier 1024、FFN 4096，BFV slot
count=8192。

padding 不是“置零后仍送入 Softmax”：QK 后先应用固定公开 valid mask，再公开切出
`197×197` secret-share 子矩阵调用 Softmax；结果装回 256 物理布局时补零，并在每个
block residual 边界重新应用公开二值 token mask。

## 5. 分阶段实测结果

### A. PyTorch / floating checkpoint reference — PASS

真实 parrots 图片的 Hugging Face PyTorch top-1 为 88，canonical floating graph
top-1 也是 88；1000 logits 最大绝对差为 `2.243422856512689e-06`。证据：
`outputs/vit_b16_pytorch_reference_current/pytorch_reference.json`。

### B. BOLT fixed-point clear reference — PARTIAL PASS

完整 12 层 exact fixed reference 的 clear/fixed top-1 都是 88；197→256 padding 的
valid-prefix max difference 为 `0.0`。BFV pre-modulus range audit 完整覆盖
`123/123` 个 accumulator，全部 PASS、无 wrap。最紧的是 `block_6.fc1`：
`peak_abs=193304523`，centered bound `268451840`，utilization
`0.7200715145033091`（约 `0.7200715`）。证据位于
`outputs/vit_b16_reference_exact_full/`。

最新 clear/fixed max-absolute error 摘要为：patch projection
`0.007109827601961216`，所有 block 中 attention residual 最大
`1486.0443772620538`，FFN residual 最大 `1513.357238896947`，final CLS
`2.392179370326475`，logits `1.9317061266129691`。top-1 与 range gate 已通过，
但仓库尚未定义逐层数值接受阈值，因此这些逐层 `status=PASS` 只表示 shape 可比较、
指标已计算，不能把“逐层数值精度”宣称为 PASS。

LayerNorm checkpoint epsilon 是 `1e-12`；在 scale=12 下量化为 0，而且现有
`FPMath::layer_norm_fix` 核心没有显式 epsilon 加法。fixed 与 secure reference
按核心真实行为记录 `layer_norm_epsilon_applied_by_core=false`，没有偷偷改变
LayerNorm 协议。这也是解释 float/fixed 差异时必须保留的限制。

### C. 单个真实 ViT block secure inference — PASS（validation block limit）

`outputs/vit_b16_secure_real_block1_aggregate_exact_v2/` 使用真实 checkpoint、真实
图片、BERT conversion/packing 复用入口和一个 block。clear、fixed、secure top-1
均为 733；patch projection、attention residual、FFN/MLP residual、CLS、logits 五个
边界的 secure/fixed max/mean/RMSE 全为 `0.0`，padding max difference 为 `0.0`。
secure/clear logits max error 为 `0.24589157104492188`。validation observer 只恢复每个
边界的聚合误差标量，不恢复中间 tensor。

该次实测性能：

| 范围 | time | bidirectional bytes | logical rounds |
|---|---:|---:|---:|
| end-to-end | 480976.32968 ms | 27761622618 | 10325 |
| online inference | 444610.288237 ms | 27721228452 | 10305 |

online backend breakdown 精确闭合：

| backend | time | bytes | rounds |
|---|---:|---:|---:|
| conversion | 85192.620 ms | 5024330754 | 193 |
| HE | 81314.140 ms | 0 | 0 |
| MPC | 192601.484 ms | 16390670410 | 9356 |
| local/public | 280.186 ms | 14033392 | 122 |
| runtime overhead / observer transport | 85221.858237 ms | 6292193896 | 634 |

该图记录 433 次 noise flooding 和 866 次 modulus switching；runner 对每个相关
HE 输出执行 `floods>0` 且 `switches>=2*floods` 的 fail-closed trace gate，主执行图
的 circuit-privacy hardening 已实际接线并由 trace 验证。

### D/E. 完整 12 层 secure inference / 真实图片 top-1 — PASS

`outputs/vit_b16_secure_real_full12_aggregate_exact/` 使用真实 checkpoint、真实
parrots 图片、12 个 block 和 validation-only 精确聚合误差 observer。两方完整结束，
clear/fixed/secure top-1 均为 88，secure/fixed 1000 logits max/mean/RMSE 均为
`0.0`，padding valid-prefix max difference 为 `0.0`。150 个 reference operator、
12 个 block 的 hardening trace 和性能账目门禁全部 PASS；secure 中间 tensor
重构次数为 0。独立 `summarize_vit_acceptance.py` 以 `secure_fixed_atol=0` 复验后
返回 `OVERALL=PASS`。

observer 覆盖 27 个边界：patch projection、每层 attention residual、每层 FFN/MLP
residual、最终 CLS 和 logits。它把 secure share 提升到 ell=43/scale=18；差值平方
保留 scale=36，以 `k=ceil(log2(N))` 拆成高低 limb，秘密求和并在秘密态处理 carry，
只公开每个 reference 的 max、sum-absolute 和规范化总平方和。公开的两个平方和
limb 与一个总平方和一一对应，不包含分桶或逐元素信息，也不恢复中间 tensor。

27 个边界的 fixed/secure max/mean/RMSE 全为 `0.0`，所以
`first_inconsistent_layer=NONE_EXACT_MATCH`。clear oracle 的量化界为
`1.9073486328125e-06`；所有 clear/secure 与 clear/fixed 的 max/mean/RMSE 差都在
该界内。逐层三路数值、shape、scale、bit length 和 packing 见
`outputs/vit_b16_secure_real_full12_aggregate_exact/layer_error_summary.csv`。其中
`shape` 是物理 tensor（block 边界为 `[256,768]`），`observed_shape` 明确记录误差
统计只覆盖有效前缀 `[197,768]`；`fixed_secure_acceptance=PASS_ATOL_0` 与没有预设
阈值的 `clear_fixed_numeric_acceptance` 分列，避免混淆。

完整运行性能如下。这里的 `runtime_overhead` 包含 27 次聚合误差观测，因此这是
验收图的开销，不是关闭 observer 后的 production latency：

| 范围 | time | bidirectional bytes | logical rounds |
|---|---:|---:|---:|
| end-to-end | 4746696.70841 ms（79.112 min） | 292024793966 | 27925 |
| online inference | 4708284.01263 ms | 291984399800 | 27905 |

online breakdown 精确闭合：

| backend | time | bytes | rounds |
|---|---:|---:|---:|
| conversion | 924432.665 ms | 54488288418 | 1865 |
| HE | 932805.907 ms | 0 | 0 |
| MPC | 2142628.549 ms | 185082986510 | 21544 |
| local/public | 452.649 ms | 14033392 | 122 |
| runtime overhead / observer transport | 707964.24263 ms | 52399091480 | 4374 |

HE 侧累计 41521 ciphertexts、616046 rotations、2747992 multiply-plain、
294912 ct-ct multiply、37152 relinearizations、9314 modulus switches 和 4657
noise floodings。证据为同目录的 `performance.csv`、`backend_breakdown.csv`、
`operator_trace.csv`、`reference_report.json` 和 `run_metadata.json`。

该已完成 artifact 的 `alice.log` 中单行 `BOLT_VIT_E2E_METRICS` 曾把
`inference_time_ms` 误标为 E2E 时间；权威值一直是 `performance.csv` 的
`online_inference=4708284.01263 ms`，源码现已改为分别输出 online 与 E2E。这个标签
修正不改变执行图或上面的原始计时 CSV。

不启用 observer 的完整 12 层基线位于 `outputs/vit_b16_secure_real_full12/`：
end-to-end `4277089.19519 ms`（71.285 min）、`239611669094` bytes、23429 rounds；
online `4225364.7052 ms`、`239571274928` bytes、23409 rounds。该组更接近 production
开销；相对该基线，observer 验收图的 E2E 时间、通信、rounds 分别增加约 10.98%、
21.87%、19.19%，与额外 27 个聚合观测边界的工作量一致。上表用于逐层验收，不能
当作关闭 observer 后的 latency。

### 多 seed 回归 — PASS（synthetic secure regression）

seed 20260810、20260811、20260812 共 `3/3` secure/fixed top-1 一致，padding
均 PASS；三次 secure/fixed logits 最大误差中的最大值为
`0.683837890625`（约 `0.6838379`）。证据：
`outputs/vit_b16_seed_regression_current/summary.csv`。

这是多 seed 协议回归 fixture，不等价于三张真实 checkpoint/ImageNet 图片的完整
12 层 secure 回归。该 synthetic 回归没有启用 observer，因此其
`first_inconsistent_layer` 仍正确记录为
`NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER`；上面的真实 full-12 observer
则给出 `NONE_EXACT_MATCH`。

### F. ImageNet-1k validation — PyTorch PASS；secure NOT RUN

本机 50,000 张 ImageNet validation 全量真实 checkpoint 评估已经完成：top-1
`40163/50000 = 80.326%`，top-5 `47743/50000 = 95.486%`；不是由小样本外推。
证据为 `outputs/vit_b16_imagenet_eval/imagenet50k.json`。命令：

```bash
../BOLT/.venv-vit-b16/bin/python vit_b16/scripts/evaluate_vit_b16_pytorch.py \
  --checkpoint-dir ../BOLT/checkpoints/vit_b16_hf \
  --manifest ../BOLT/datasets/imagenet2012/val/validation_manifest.csv \
  --image-root ../BOLT/datasets/imagenet2012/val \
  --output outputs/vit_b16_imagenet_eval/imagenet50k.json \
  --batch-size 32
```

50,000 张 secure accuracy 尚未运行，不能用 PyTorch accuracy 替代。

## 6. 安全边界与当前验收表

production 只恢复最终 class；普通 validation 最多额外恢复最终 logits。两者都不
恢复 patch、block、MLP、CLS 等 secure 中间 tensor，不使用零随机 mask、明文算子、
安全占位符或跳层。

显式设置 `AGGREGATE_ERRORS=1` 才启用 validation-only observer。它对每个边界在
secret-share 域计算 max absolute error、sum absolute error 和 exact sum squared
error。每个边界公开 2 个 max、2 个 sum-absolute 和平方和的 4 个 high/low limb
ring words；这些信息等价于 fixed/secure 与 clear/secure 各自的六个
max/mean/RMSE 统计量，不包含逐元素值。完整 12 层共 27 个边界、没有中间 tensor
reconstruction；独立
schema/invariant validator 还验证 `mean <= RMSE <= max`、元素计数、两-limb carry、
scale 和边界顺序。oracle share 在本机复现实验中为方便运行而同机存放，但每个进程
只读取自己的 share；真实双机部署必须把两方 provisioning 文件物理隔离。observer
不编入 production 数据流，其结果不能用来降低 production 的隐私边界。

正式复现入口 `repro/run_vit_b16.sh` 在启动两方前强制校验 checkpoint manifest 和
raw/canonical SHA。底层 binary 本身只读取 canonical model、没有 manifest 参数；若
绕过 runner 直接调用 binary，checkpoint provenance 不再 fail-closed，因此这种调用
不属于本文的真实 checkpoint 验收范围。

| 验收项 | 状态 | 证据/缺口 |
|---|---|---|
| BERT 专用硬编码审计、runtime-shape ViT 适配 | PASS | 见第 1 节；BERT 文件未被改写成 ViT |
| 真实 checkpoint SHA、strict loader、fused/independent QKV | PASS | raw/canonical SHA 与全部 tensor consumption 已验证 |
| A：PyTorch ↔ canonical floating | PASS | top-1 88；logits max abs `2.243422856512689e-06` |
| B：fixed top-1 / padding invariance | PASS | top-1 88；valid-prefix diff 0 |
| B：BFV pre-modulus range | PASS | 123/123；worst utilization `0.7200715` |
| B：逐层 clear/fixed 数值阈值 | NOT EVALUATED / THRESHOLD NOT DEFINED | 已报告每层误差；不把缺少阈值误报为数值精度 PASS |
| encrypted T=256 packing / conversion / linear / attention / Softmax unit | PASS | `outputs/vit_b16_cpp_protocol_tests_current/summary.csv`；全部两方测试 exit 0 |
| circuit-privacy hardening 主执行图 | PASS | full-12 为 4657 flooding / 9314 mod-switch；逐算子 trace gate PASS |
| C：真实 checkpoint 单 block secure | PASS | 五个聚合边界 exact match；三路 top-1 733 |
| D：完整真实 12 层 secure | PASS | 12/12 blocks、150 reference operators、secure 图两方完整结束 |
| E：真实图片 secure top-1 | PASS | clear/fixed/secure 均为 88；padding diff 0 |
| 三 seed secure regression | PASS（synthetic） | 3/3 top-1；max logits error `0.6838379` |
| 真实 checkpoint、多真实图片的 full-12 多 seed 回归 | NOT RUN | synthetic 3-seed 不能替代该项 |
| secure 中间逐层聚合误差 | PASS | 27/27 边界 fixed/secure max/mean/RMSE 全为 0；未恢复 tensor |
| secure 首个不一致层 | PASS | `NONE_EXACT_MATCH` |
| 本机 validation oracle share 的物理角色隔离 | **FAIL / DEV-ONLY** | 两方同 UID、share 文件同目录；代码只读本方文件，但真实双机必须分别 provision |
| 一层及完整 12 层时间/通信与 backend breakdown | PASS | 第 5 节；各分项与 online totals 精确闭合 |
| ImageNet 50k PyTorch accuracy | PASS | top-1 80.326%，top-5 95.486% |
| ImageNet 50k fixed-point accuracy | NOT RUN | 不以单图 top-1 或 PyTorch accuracy 代替 |
| ImageNet 50k secure accuracy | NOT RUN | 不虚构、不由 PyTorch 数值外推 |
| 原 BERT 正式路径静态审计 | PASS | `repro/audit_formal_blb.sh` |
| 原 BERT 动态回归 | **FAIL / BLOCKED** | 仓库既有正式 BERT 状态为 `formal_ready=no`；诊断运行未产出可接受 oracle MSE，不能声称原 BERT tests PASS |
| 密码学协议数学定义与安全模型不变 | PASS | ViT 只调用既有 BFV/MPC/OT/转换/非线性/MatMul/packing；两处 SCI 差异见 1.4，不改变公式 |
| “只新增 ViT 文件”的字节级源码条件 | **FAIL / QUALIFIED** | 恢复 LinearOT 两 cross-term 并行 pragma、删除 secret-share 调试打印；不是协议改写，但核心源码并非逐字节不变 |

`outputs/vit_b16_acceptance_current.json` 对本次有数据的 ViT scope 返回
`OVERALL=PASS`：真实 checkpoint、A/B/C/D/E、BFV range、完整 12 层安全图、27 个
聚合误差边界、性能闭合和 ImageNet PyTorch gate 均通过。这个结论不扩张为所有原始
硬性要求的无条件 PASS：原 BERT 动态回归仍 BLOCKED，真实 checkpoint 的 full-12
多图片/多 seed 与 50k fixed/secure accuracy 尚未运行，clear/fixed 逐层误差也没有
预先定义的数值阈值，而且本机 observer 复现实验没有提供两方文件系统物理隔离。
未运行项不以随机权重、单图或明文结果替代。
