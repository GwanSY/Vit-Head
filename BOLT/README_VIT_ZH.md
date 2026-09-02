# BOLT BERT / ViT 复现（CPU）

所有命令都在 `BOLT/` 目录执行。BERT 必须先跑通，再运行 ViT。

## 2. 目录结构

```text
BOLT/
├── SCI/                         # SCI、SEAL 和 BOLT 源码
├── SCI/build_vit/bin/           # 编译后的程序
├── datasets/.../quantize/mrpc/  # BERT 数据目录（下面直接有 weights_txt）
├── fixtures/vit_smoke/           # Tiny ViT fixture
├── checkpoints/                 # ViT checkpoint/转换结果
├── outputs/                     # 日志、trace、结果
└── scripts/                     # 构建、预检、运行脚本
```

BERT 的 `<data>` 必须是直接包含 `weights_txt/` 的目录，例如：

```text
/home/dm/mind/Vit-Head/BOLT/datasets/official_bert/bolt/quantize/mrpc
└── weights_txt/
    ├── bert.encoder.layer.0.attention.self.query.weight.txt
    ├── classifier.weight.txt
    ├── inputs_0_data.txt
    └── inputs_0_mask.txt
```

## 3. 构建

```bash
sudo apt-get install -y g++ cmake make git libgmp-dev libssl-dev iproute2
JOBS=4 TARGET=BOLT_BERT scripts/build.sh
JOBS=4 TARGET=BOLT_VIT scripts/build.sh
```

构建目录固定为 `SCI/build_vit/`，不使用其他项目的 build 目录。

```bash
python3 -m venv .venv-vit-b16
.venv-vit-b16/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1 torchvision==0.22.1
.venv-vit-b16/bin/python -m pip install \
  --index-url https://pypi.org/simple \
  -r requirements-vit-b16.txt
```

当前工作区已经准备好 `.venv-vit-b16/`。

构建后可用以下命令检查 strict loader 的 fused/independent QKV 两种格式：

```bash
python3 scripts/test_vit_checkpoint_loader.py
```

这是 deterministic schema 单测；它会明确输出
`CHECKPOINT_INTEGRATION=NOT_RUN_NO_REAL_VIT_B16_CHECKPOINT`，不会把合成张量
冒充真实 checkpoint。

## 4. 启动 BERT-12 层

官方 `bolt.zip` 解压后的 MRPC 数据路径：

```bash
BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc"
```

先做数据预检：

```bash
cd /home/dm/mind/Vit-Head/BOLT

python3 scripts/preflight_bert_data.py \
  --data "$PWD/datasets/official_bert/bolt/quantize/mrpc" \
  --role both --id 0 --num-sample 1
```

### 4.1 不限速运行

```bash
mkdir -p outputs/bert_e2e_mrpc
BERT_DATA="$BERT_DATA" \
OUTPUT="$PWD/outputs/bert_e2e_mrpc" \
PORT=18000 SAMPLE_ID=0 NUM_SAMPLE=1 \
  scripts/run_bert_e2e.sh 2>&1 | tee outputs/bert_e2e_mrpc/runner.log
```
#### 改显卡
cd /home/dm/mind/Vit-Head/BOLT

unset BOLT_VIT_MAX_LAYERS

CUDA_VISIBLE_DEVICES=0 \
BINARY="$PWD/SCI/build_gpu/bin/BOLT_VIT_GPU" \
scripts/run_vit_gpu.sh \
  "$PWD/../datasets/vit_b16_shared/images/image_000.png" \
  "$PWD/outputs/accuracy_secure_smoke" \
  49100
  
同机运行使用端口 `18000–18064`；完整 BERT 内存占用很高，建议至少 160 GB
可用内存。

### 4.2 100 Mbps、80 ms RTT 运行

需要 root 或 `CAP_NET_ADMIN`。`RTT_MS=80` 会在两个方向各加入 40 ms 延迟。

```bash
sudo env IFACE=lo RATE_MBPS=100 RTT_MS=80 \
  scripts/run_with_network_profile.sh -- \
  env BERT_DATA="$BERT_DATA" \
  OUTPUT="$PWD/outputs/bert_e2e_100m_80ms" \
  PORT=18000 SAMPLE_ID=0 NUM_SAMPLE=1 \
  scripts/run_bert_e2e.sh
```

网络仿真只改变网卡上的速率和延迟，不改变 BFV/MPC 参数；进程退出后 wrapper
会删除临时 `tc` qdisc。

### 4.3 查看进度

另开终端：

```bash
tail -f outputs/bert_e2e_mrpc/alice.log
tail -f outputs/bert_e2e_mrpc/bob.log
```

日志应依次出现 `Layer - 0` 到 `Layer - 11`。`results.csv` 在最终 reveal 前为
空文件是正常的。

### 4.4 BERT 完成检查

```bash
test -s outputs/bert_e2e_mrpc/results.csv
grep -q '^BOLT_BERT_E2E=PASS$' outputs/bert_e2e_mrpc/runner.log
grep -q '^BOLT_BERT_E2E_METRICS ' outputs/bert_e2e_mrpc/alice.log
cat outputs/bert_e2e_mrpc/results.csv
```

日志中的指标格式：

```text
BOLT_BERT_E2E_METRICS communication_bytes=... end_to_end_time_ms=... logical_rounds=... inference_communication_bytes=... inference_time_ms=... status=PASS
```

其中 `communication_bytes` 为双方 send-only counter 的双向和，`logical_rounds`
和时间取双方最大值；指标交换自身不计入推理区间。

然后运行共享后端回归：

```bash
JOBS=4 TARGET=BERT_SHARED_BACKEND_REGRESSION scripts/build_vit.sh
scripts/run_bert_shared_backend_regression.sh
cat outputs/regression/bert_shared_backend.csv
```

## 5. 运行 Tiny ViT

### 5.1 单层

```bash
mkdir -p outputs/vit_smoke
OUTPUT="$PWD/outputs/vit_smoke" PORT=19000 \
  scripts/run_vit_e2e.sh 2>&1 | tee outputs/vit_smoke/runner.log
```

### 5.2 两层

```bash
python3 scripts/generate_vit_fixture.py \
  --out-dir fixtures/vit_smoke_multilayer --layers 2 --seed 20260804
FIXTURE="$PWD/fixtures/vit_smoke_multilayer" \
OUTPUT="$PWD/outputs/vit_smoke_multilayer" PORT=19001 \
  scripts/run_vit_e2e.sh
```

### 5.3 ViT 检查

```bash
test -f outputs/vit_smoke/PASS
test -s outputs/vit_smoke/prediction.txt
grep -q '^BOLT_VIT_E2E_METRICS ' outputs/vit_smoke/alice.log
grep -q '^PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0$' outputs/vit_smoke/alice.log
grep -q '^SCALAR_FALLBACKS=0$' outputs/vit_smoke/alice.log
grep -q '^GENERIC_OT_MATMUL_CALLS=0$' outputs/vit_smoke/alice.log
cat outputs/vit_smoke/prediction.txt
```

`performance.csv` 中的 `end_to_end_inference` 行与 ALICE 日志的端到端通信量、
时间和逻辑轮数一致。`run_metadata.json` 会记录实际网络 profile。

## 6. 标准 ViT-B/16：转换、reference 与两方推理

当前工作区已准备官方 `google/vit-base-patch16-224` checkpoint：

```text
checkpoints/vit_b16_hf/model.safetensors
checkpoints/vit_b16/bolt_model.txt
checkpoints/vit_b16/manifest.json
```

权重 SHA256 为
`1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d`，来源 revision
为 `10064cc17b69d623a025c9e90e11230cc8ba6967`。如需重新准备，先固定原始
SHA256 并转换：

```bash
python3 scripts/convert_vit_checkpoint.py \
  --checkpoint /secure/path/pytorch_model.safetensors \
  --config /secure/path/config.json \
  --preprocessor /secure/path/preprocessor_config.json \
  --output checkpoints/vit_b16/bolt_model.txt \
  --manifest checkpoints/vit_b16/manifest.json \
  --source-id google/vit-base-patch16-224 \
  --source-revision <pinned-revision> \
  --expected-sha256 <sha256>
```

没有 `--allow-non-b16` 时，loader 必须检查
`224/16/3/197/768/12/3072/12/1000`；缺 tensor、形状错误、非 pre-LN 或
checkpoint hash 不匹配都会失败。

PyTorch reference 环境已经固定在 `.venv-vit-b16/`。验证导入：

```bash
.venv-vit-b16/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

单张真实图片的 PyTorch/reference 对齐：

```bash
.venv-vit-b16/bin/python scripts/vit_pytorch_reference.py \
  --checkpoint checkpoints/vit_b16_hf/model.safetensors \
  --config checkpoints/vit_b16_hf/config.json \
  --preprocessor checkpoints/vit_b16_hf/preprocessor_config.json \
  --image /path/to/image.jpg \
  --output outputs/vit_b16_e2e/pytorch_reference.json \
  --expected-sha256 1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d
```

先做 A/B reference：

```bash
python3 scripts/vit_pytorch_reference.py \
  --checkpoint /secure/path/pytorch_model.safetensors \
  --config /secure/path/config.json \
  --preprocessor /secure/path/preprocessor_config.json \
  --image /path/to/image.jpg \
  --output outputs/vit_b16_e2e/pytorch_reference.json \
  --expected-sha256 <sha256>
python3 scripts/preprocess_vit_image.py /path/to/image.jpg \
  --manifest checkpoints/vit_b16/manifest.json \
  --output outputs/vit_b16_e2e/bob_image.txt
python3 scripts/vit_reference.py \
  --model checkpoints/vit_b16/bolt_model.txt \
  --input outputs/vit_b16_e2e/bob_image.txt \
  --output outputs/vit_b16_e2e/reference_report.json
python3 scripts/compare_vit_logits.py \
  --pytorch outputs/vit_b16_e2e/pytorch_reference.json \
  --canonical outputs/vit_b16_e2e/reference_report.json
```

本机若未安装 PyTorch，第一条命令会明确报 dependency-gated failure；不能用
NumPy reference 的 PASS 替代这个阶段。

再启动完整两方 secure 图：

```bash
MODEL="$PWD/checkpoints/vit_b16/bolt_model.txt" \
IMAGE=/path/to/image.jpg \
MANIFEST="$PWD/checkpoints/vit_b16/manifest.json" \
OUTPUT="$PWD/outputs/vit_b16_e2e" PORT=49000 \
  scripts/run_vit_b16_e2e.sh
```

如果使用当前工作区已准备的模型，可直接运行：

```bash
scripts/run_prepared_vit_b16.sh /path/to/image.jpg
```

当前工作区可用断点安全的官方 validation 下载器。它按磁盘实际长度发起新的
Range 请求，且只在字节数、官方 MD5 和 50,000 个 tar member 全部正确后落盘：

```bash
scripts/download_imagenet_val.sh datasets/imagenet2012
```

ImageNet validation 原始 tar 和官方 devkit 放在 `datasets/imagenet2012/` 后，
生成图片目录与零基标签 manifest：

```bash
.venv-vit-b16/bin/python scripts/prepare_imagenet_val.py \
  --val-tar datasets/imagenet2012/ILSVRC2012_img_val.tar \
  --devkit datasets/imagenet2012/ILSVRC2012_devkit_t12.tar.gz \
  --output-root datasets/imagenet2012/val
```

标签不是把 devkit ground-truth 数字简单减一；脚本通过官方 `meta.mat` 执行
`ILSVRC2012 synset id -> leaf WNID -> 标准有序 WNID 类别索引` 映射，并输出
`class_index.json` 供审计。

PyTorch 全量/子集 accuracy：

```bash
.venv-vit-b16/bin/python scripts/evaluate_vit_b16_pytorch.py \
  --checkpoint-dir checkpoints/vit_b16_hf \
  --manifest datasets/imagenet2012/val/validation_manifest.csv \
  --image-root datasets/imagenet2012/val \
  --output outputs/accuracy/vit_b16_pytorch.json \
  --batch-size 16
```

脚本只把 canonical model 作为 ALICE 参数传入，BOB 只得到本地预处理图片；
`debug_logits=0` 固定启用，最终只 reveal class index。输出目录包括：

- `operator_trace.csv`：每个 patch/attention/MLP/LN/CLS/logit 算子的 shape、
  scale、ell、packing、HE operation counts、bytes 和 rounds；
- `performance.csv`：BFV context/key、evaluation keys、base OT、online HE/MPC/
  conversion 混合阶段和 final reveal 的时间/通信 breakdown；
- `backend_breakdown.csv`：adapter trace 按 `he`、`mpc`、`conversion` 和
  `local_or_public` 分组的时间、双向通信和操作计数；时间是逐算子行汇总，
  不是把两端 wall-clock 相加；
- `run_metadata.json`：logical/physical tokens、BFV 参数、固定点和 fail-closed
  标志；
- `reference_report.json`：clear/fixed clear 误差及 padding invariance。secure
  中间误差在没有 aggregate-only validation observer 时必须是
  `NOT_AVAILABLE`，不能使用解密中间值替代。

## 7. 密码后端

| 算子 | BOLT 后端 |
|---|---|
| Patch projection、Q/K/V、输出投影、FC1/FC2、分类头 | BFV ciphertext-plaintext compact MatMul + BSGS |
| QKᵀ、Softmax×V | BFV ciphertext-ciphertext cross packing + `Evaluator::multiply` + relinearization |
| Softmax、LayerNorm、GELU | 原 BOLT/SCI MPC 定点实现 |
| HE↔MPC | BOLT prime-field / `Z2^ell` conversion |
| residual、CLS、position、patchify | 本地 share 操作，不通信 |

HE 使用 SEAL **BFV**，不是 CKKS；主参数为 poly degree 8192、plaintext modulus
`536903681`、`Z2^37` share、activation scale 12。正式路径禁止 scalar
ciphertext、旧 `BfvLinear`、`LinearOT::matrix_multiplication`、明文中间重构和
静默 fallback。

## 8. 当前边界

已通过：BERT shared backend、ViT 三内核、Attention 协议测试、Stem/LN1 测试、
Tiny 单层和 Tiny 多层运行。

官方 ViT-B/16 checkpoint、PyTorch reference 和 50,000 张 ImageNet validation
数据均已准备。16 张真实 validation 烟测为 top-1 `81.25%`、top-5 `100%`；这只
验证数据/环境链路，不能作为全量 accuracy。当前公开图片 smoke 中，PyTorch 与
canonical clear top-1 一致，但 fixed-clear top-1 仍需单独验收，不能把 PyTorch
PASS 自动当作 secure/fixed PASS。
当前精确验收状态和 BERT 硬编码审计见
[`docs/bolt_vit_b16_audit_and_status.md`](docs/bolt_vit_b16_audit_and_status.md)。
