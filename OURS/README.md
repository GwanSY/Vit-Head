# BLB：BERT 论文复现与运行

本 README 先给出可直接复制的 BERT/MRPC 命令。ViT-B/16 代码仍保留在仓库中，
但它是独立适配路径，不作为当前 BERT 论文复现已经完成的证据。

## 当前结论

| 项目 | 当前状态 | 结论 |
|---|---|---|
| 真实 checkpoint、12 层 fixed-clear、408 个 MRPC 样本 | **PASS** | `acc_and_f1=88.4437%`；论文为 `88.5%`，差 `-0.0563` 个百分点 |
| formal BERT CPU/SEAL | 可运行 layer 0 | 只验收单层安全执行图，不是完整 BERT 准确率 |
| formal BERT GPU/Phantom | 可运行 layer 0 | 真实 CKKS plaintext 的 device 镜像已经接入；用 `BACKEND=gpu` 选择 |
| formal 12 层循环、pooler、classifier | **未接通** | 入口会拒绝 `layers != 1`，不能声明 secure 论文复现 PASS |

`88.4437%` 是完整真实权重的**定点明文参考**结果，不是两方安全推理结果。
build PASS、manifest PASS 或单层 PASS 也都不能替代完整 12 层 secure 验收。

## 1. 进入环境

在宿主机执行；脚本会创建、启动或复用 `blb_formal_env_fix`，进入后目录为
`/workspace/BLB`：

```bash
cd /home/dm/mind/MIND_code/Pencil/BLB
CUDA_ROOT=/usr/local/cuda-12.3 \
BLB_CONTAINER_NAME=blb_formal_env_fix \
bash repro/run_official_container.sh
```

容器需要可用的 Docker、NVIDIA runtime 和宿主机 CUDA toolkit。若当前用户无
Docker 权限，先按本机配置加入 `docker` 组；不要用宿主机构建产物冒充容器构建产物。

## 2. 编译 BERT formal 目标

以下命令在容器内执行。A100 使用 `CUDA_ARCH=80`；`CUDA_ARCH` 是编译架构，
不是 GPU 编号。一次 `USE_HE_GPU=ON` 构建同时支持运行时 CPU/SEAL 和 GPU/Phantom：

```bash
cd /workspace/BLB
CUDA_ROOT=/usr/local/cuda-12.3 \
CUDA_ARCH=80 \
USE_HE_GPU=ON \
JOBS=2 \
BUILD_ROOT=/workspace/BLB/build_blb_formal \
bash repro/build_formal_blb.sh
```

构建产物：

```text
/workspace/BLB/build_blb_formal/sci/bin/ckks_bert_large_main
```

无 NVIDIA GPU 时可用 `USE_HE_GPU=OFF` 构建 CPU-only 目标。常见架构示例：V100
为 `70`，A100 为 `80`，RTX 30 系列为 `86`，RTX 40 系列为 `89`。

## 3. 先跑完整 fixed-clear 基线

该命令使用真实 12 层权重、pooler/classifier 和全部 408 个 MRPC 样本：

```bash
cd /workspace/BLB
python3 repro/run_bert_fixed_clear.py \
  --data-dir checkpoints/official-blb-mrpc/weights_txt \
  --samples 408 \
  --paper-target 88.5 \
  --paper-tolerance-pp 0.1 \
  --json-out outputs/bert_fixed_clear_408.json
```

当前仓库真实资产的记录值：

| Metric | 当前结果 | 论文 Table 8 |
|---|---:|---:|
| Accuracy | 86.5196% | — |
| F1 | 90.3678% | — |
| MRPC `acc_and_f1` | **88.4437%** | **88.5%** |

快速检查一个样本：

```bash
python3 repro/run_bert_fixed_clear.py --samples 1 --progress-every 1
```

## 4. 运行 formal BERT layer 0

先检查真实权重 manifest 和正式路径静态门禁：

```bash
cd /workspace/BLB
python3 repro/verify_bert_manifest.py \
  checkpoints/official-blb-mrpc/weights_txt --layer 0
bash repro/audit_formal_blb.sh
```

### CPU / Microsoft SEAL

```bash
cd /workspace/BLB
BACKEND=cpu \
BUILD_DIR=/workspace/BLB/build_blb_formal/sci \
MODEL_PATH=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
OUTPUT=/workspace/BLB/outputs/bert_layer0_cpu \
PORT=62000 \
RUN_TIMEOUT=0 \
bash repro/run_original_bert.sh
```

### GPU / PhantomFHE

下面选择第 1 张可见 GPU；改成 `GPU_ID=1` 即选择第 2 张：

```bash
cd /workspace/BLB
BACKEND=gpu \
GPU_ID=0 \
BUILD_DIR=/workspace/BLB/build_blb_formal/sci \
MODEL_PATH=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
OUTPUT=/workspace/BLB/outputs/bert_layer0_gpu \
PORT=62100 \
RUN_TIMEOUT=0 \
bash repro/run_original_bert.sh
```

确认实际进入 GPU 后端并上传了真实 plaintext：

```bash
grep -E 'HE_BACKEND mode=gpu|CKKS_DEVICE_PLAIN_CACHE' \
  outputs/bert_layer0_gpu/alice.log
```

这里无需准备一份“GPU 权重”。CPU 与 GPU 读取同一组真实 BERT txt 权重；
`SCI/tests/bert_large_bolt/ckks_plain_cache.{h,cpp}` 会按算子要求的 CKKS
chain level 和 scale，把 host `seal::Plaintext` 物化并上传为 Phantom device
plaintext，算子结束后清理 device cache。

`GPU_ID`/`CUDA_VISIBLE_DEVICES` 选择显卡，`BACKEND=gpu` 才选择 Phantom evaluator。
GPU 只加速适合的 HE 线性部分，MPC、OT、HE↔MPC 转换仍走原协议，因此不能只凭
GPU 利用率判断协议正确性或预期加速比。

上述 CPU/GPU 命令当前都只执行 formal layer 0。脚本成功输出
`SINGLE_LAYER_FORMAL status=PASS` 也只代表单层验收；12 层循环和分类头尚未接通，
所以当前不得写成“secure BERT 已复现论文 88.5%”。详细阻塞项见
[REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

## 5. 100 Mbps、80 ms RTT

容器内 loopback 限速、CPU/GPU 顺序运行和恢复网络的完整复制命令见：

[repro/README_100M_80MS.md](repro/README_100M_80MS.md)

该配置使用 `100 mbit` 和每方向 `40 ms`，约等于 `80 ms RTT`。论文完整
BERT-base WAN3 数据只能作为目标值；当前 formal 入口只有 layer 0，不能直接与论文
完整 12 层端到端时间比较。

## 6. 文件与目录

| 路径 | 功能 |
|---|---|
| `SCI/` | BLB 原有 BFV/CKKS、MPC、OT、HE↔MPC 和非线性协议实现 |
| `SCI/tests/bert_large_bolt/` | BERT/BLB 执行图、CKKS 线性算子及 formal main |
| `SCI/tests/bert_large_bolt/ckks_plain_cache.{h,cpp}` | 真实 CKKS host plaintext 到 Phantom device plaintext 的 level/scale 对齐缓存 |
| `repro/` | 环境、构建、运行、manifest、审计和验收脚本 |
| `repro/run_official_container.sh` | 创建/复用官方 BLB 环境并挂载当前仓库 |
| `repro/build_formal_blb.sh` | 构建 formal BERT CPU/GPU 目标 |
| `repro/run_original_bert.sh` | 启动 ALICE/BOB formal layer-0 CPU 或 GPU 运行 |
| `repro/run_bert_fixed_clear.py` | 真实 12 层 BERT/MRPC 定点明文参考 |
| `repro/README_100M_80MS.md` | 100 Mbps、80 ms RTT 的 BERT CPU/GPU 命令 |
| `checkpoints/official-blb-mrpc/weights_txt/` | 真实 BERT/MRPC 权重、输入、mask、oracle shares 和 manifest |
| `build_blb_formal/` | formal BERT 构建产物，生成目录 |
| `outputs/` | 运行日志、JSON 结果和验收输出 |
| `REPRODUCTION_STATUS.md` | formal BERT 当前 PASS/BLOCKED 证据 |
| `vit_b16/scripts/head_hetero_calibrate.py` | 头级异构协议离线校准 |
| `vit_b16/scripts/head_hetero_reorder.py` | 头级异构权重重排与审计 |

## 7. 头级异构 ViT

OURS 的正式 ViT 方法只保留头级异构路径。`vit_b16/scripts/` 中的校准和重排工具
用于在公开校准图片上选择 `softmax_mpc`、`relu_norm_mpc` 与 `he_linear`，不再提供
独立的 OURS-B16-compatible 构建/运行入口。
