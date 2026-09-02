# BOLT 快速运行

以下命令都在 BOLT 根目录执行：

```bash
cd /home/dm/mind/Vit-Head/BOLT
```

## 1. 先选择模型，再选择执行后端

模型和执行后端是两个不同维度：

- 模型决定运行 BERT 还是 ViT-B/16；
- 执行后端决定密码协议在 CPU 还是 GPU 上执行。

### 1.1 模型选择

| 模型 | CMake target | 编译后的程序 | 运行脚本 |
|---|---|---|---|
| BERT-Base（12 层） | `BOLT_BERT` | `SCI/build_vit/bin/BOLT_BERT` | `scripts/run_bert_e2e.sh` |
| ViT-B/16（12 层） | `BOLT_VIT` | `SCI/build_vit/bin/BOLT_VIT` | `scripts/run_prepared_vit_b16.sh` |

### 1.2 执行后端选择

| 执行后端 | BERT secure | ViT secure | 当前状态 |
|---|---|---|---|
| CPU | 支持 | 支持 | 当前正式后端 |
| GPU / CUDA | `BOLT_BERT_GPU` | `BOLT_VIT_GPU` | BOLT BFV + BLB Phantom 底层 evaluator |

所以当前实际可运行的组合只有：

| 模型 | 后端 | 是否可运行 | 构建 target |
|---|---|---|---|
| BERT | CPU | 是 | `BOLT_BERT` |
| ViT-B/16 | CPU | 是 | `BOLT_VIT` |
| BERT | GPU | 是（需 CUDA 驱动） | `BOLT_BERT_GPU` |
| ViT-B/16 | GPU | 是（需 CUDA 驱动） | `BOLT_VIT_GPU` |

`BOLT_BERT` 和 `BOLT_VIT` 仍是原来的 CPU/SEAL 路径。GPU 目标只把 BOLT 原有
BFV 密文算子送入 BLB 的 Phantom 底层实现，再转回 SEAL 格式；不使用 BLB 的
模型协议、CKKS 或 FineGrainFusion。IRON 的目标和代码不改，`IRON_VIT` 仍走原来的
CPU/SEAL evaluator。

### 1.3 进入 CPU 环境

```bash
cd /home/dm/mind/Vit-Head/BOLT
source .venv-vit-b16/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

当前准备好的环境应输出类似：

```text
2.7.1+cpu False
```

这个虚拟环境只负责图片预处理、checkpoint 和 PyTorch reference；C++ secure
程序仍由系统编译器运行，不依赖 CUDA。

退出环境：

```bash
deactivate
```

### 1.4 GPU secure（只用 0 号卡）

先确保 BLB GPU 构建已经生成 Phantom 包；本仓库默认查找：
`../BLB/build_vit_b16_gpu/sci/extern/phantom/lib/cmake/phantom/PhantomConfig.cmake`。
若不存在，先执行：

```bash
USE_HE_GPU=ON BLB_VIT_USE_GPU=ON \
  BUILD_ROOT="$PWD/../BLB/build_vit_b16_gpu" \
  bash ../BLB/repro/build_vit_b16.sh
```

构建 BOLT GPU 入口：

```bash
TARGET=BOLT_BERT_GPU JOBS=4 scripts/build.sh
TARGET=BOLT_VIT_GPU JOBS=4 scripts/build.sh
```

运行 BERT（`BERT_DATA` 指向直接包含 `weights_txt/` 的目录）：

```bash
BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc" \
OUTPUT="$PWD/outputs/bert_gpu_e2e" CUDA_VISIBLE_DEVICES=0 \
  scripts/run_bert_gpu.sh
```

运行 ViT（参数依次为图片、输出目录、端口）：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_vit_gpu.sh \
  outputs/vit_b16_smoke/parrots.png outputs/vit_b16_gpu 49000
```

GPU 只在 ALICE 的 BOLT evaluator 上执行；BOB 仍是 BOLT 原有密钥/2PC 角色。
脚本会检查 `BOLT_HE_BACKEND backend=phantom_bfv device=0 protocol=BOLT`。

[BLB 论文](https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf)
Table 9 的 BERT-base、128-token 参考值中，BOLT GPU 端到端延迟为
LAN 15.5 分钟、WAN2 77.9 分钟、WAN3 122.7 分钟。它们是论文网络配置下的
端到端值，不是本机 loopback 的硬阈值；本地复现首先应满足结果 `PASS`，再比较
同一网络配置下的时间。

## 2. 首次构建

安装系统依赖并初始化子模块：

```bash
sudo apt-get install -y \
  g++ cmake make git libgmp-dev libssl-dev iproute2 \
  python3-venv python3-pip
git submodule update --init --recursive
```

`scripts/build.sh` 是统一构建入口，真正选择模型的是 `TARGET`：

```text
TARGET=BOLT_BERT  -> 构建 BERT
TARGET=BOLT_VIT   -> 构建 ViT-B/16
TARGET=BOLT_BERT_GPU / BOLT_VIT_GPU -> 构建 BOLT GPU 入口
```

构建 BERT：

```bash
JOBS=4 TARGET=BOLT_BERT scripts/build.sh
```

构建 ViT：

```bash
JOBS=4 TARGET=BOLT_VIT scripts/build.sh
```

`scripts/build.sh` 会转调原有的共享实现 `scripts/build_vit.sh`。后者是历史命名，
继续保留以兼容已有命令；新命令不再需要把 BERT 写成 `build_vit.sh`。
构建目录名 `SCI/build_vit/` 也是同一历史原因；现在 BERT 和 ViT 共用该目录，
但会生成不同的二进制程序。

首次执行会构建 SEAL、Eigen 和 SCI。程序生成在：

```text
SCI/build_vit/bin/BOLT_VIT
SCI/build_vit/bin/BOLT_BERT
```

## 3. 修改代码后的增量编译

只重新编译 ViT：

```bash
cmake --build SCI/build_vit --target BOLT_VIT --parallel 4
```

只重新编译 BERT：

```bash
cmake --build SCI/build_vit --target BOLT_BERT --parallel 4
```

如果 CMake 配置、依赖路径或子模块发生变化，重新运行
`scripts/build.sh`，不要复用其他项目的 build 目录。

## 4. 运行 ViT-B/16（CPU secure）

使用准备好的示例图片：

```bash
scripts/run_prepared_vit_b16.sh \
  outputs/vit_b16_smoke/parrots.png \
  outputs/vit_b16_run \
  49000
```

使用真实 ImageNet validation 图片：

```bash
scripts/run_prepared_vit_b16.sh \
  datasets/imagenet2012/val/images/ILSVRC2012_val_00000001.JPEG \
  outputs/vit_b16_imagenet_00000001 \
  49001
```

三个位置参数依次是：`图片路径`、`输出目录`、`TCP 端口`。脚本会在同一台机器
启动 ALICE 和 BOB。真实 checkpoint 解析通常需要几十秒，之后每 30 秒输出一次
运行心跳。

查看日志：

```bash
tail -f outputs/vit_b16_run/alice.log
tail -f outputs/vit_b16_run/bob.log
```

## 5. 运行 BERT（CPU secure）

先确认 MRPC 数据路径；`--data` 指向 `mrpc/`，不是仓库根目录或
`weights_txt/`：

```bash
export BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc"
test -d "$BERT_DATA/weights_txt"
python3 scripts/preflight_bert_data.py \
  --data "$BERT_DATA" --role both --id 0 --num-sample 1
```

预检通过后运行：

```bash
BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc" \
OUTPUT="$PWD/outputs/bert_e2e_mrpc" \
PORT=18000 SAMPLE_ID=0 NUM_SAMPLE=1 \
  scripts/run_bert_e2e.sh
```

BERT 会使用 `PORT` 开始的一组端口；同机运行建议使用 `18000`。完整 BERT
内存占用很高，建议至少准备 160 GB 可用内存。

## 6. PyTorch CPU reference / accuracy

16 张 ImageNet 烟测：

```bash
python scripts/evaluate_vit_b16_pytorch.py \
  --checkpoint-dir checkpoints/vit_b16_hf \
  --manifest datasets/imagenet2012/val/validation_manifest.csv \
  --image-root datasets/imagenet2012/val \
  --output outputs/accuracy/vit_b16_pytorch_smoke16.json \
  --batch-size 4 \
  --max-samples 16
```

删除 `--max-samples 16` 即运行全部 50,000 张 validation 图片。

## 7. 文件夹和文件功能

| 路径 | 类型 | 功能 |
|---|---|---|
| `.venv-vit-b16/` | 文件夹 | ViT PyTorch CPU 环境 |
| `SCI/src/BOLTTransformer/` | 文件夹 | BOLT 的 BFV、packing、线性层、Attention 和 HE↔MPC 后端 |
| `SCI/tests/bert_bolt/` | 文件夹 | BERT 两方执行入口和回归测试 |
| `SCI/tests/vit_bolt/` | 文件夹 | ViT 模型适配层、secure graph、loader 和协议测试 |
| `SCI/build_vit/bin/` | 文件夹 | 编译后的 `BOLT_BERT`、`BOLT_VIT` 等程序 |
| `checkpoints/vit_b16_hf/` | 文件夹 | 原始 Hugging Face ViT-B/16 checkpoint 和配置 |
| `checkpoints/vit_b16/` | 文件夹 | 转换后的 BOLT checkpoint 与 manifest |
| `datasets/official_bert/` | 文件夹 | BERT 权重和输入数据 |
| `datasets/imagenet2012/` | 文件夹 | ImageNet validation tar、图片、标签和 manifest |
| `fixtures/` | 文件夹 | Tiny ViT 与多 seed 协议回归输入 |
| `outputs/` | 文件夹 | 推理结果、日志、性能和通信统计 |
| `scripts/` | 文件夹 | 构建、运行、checkpoint、reference 和数据工具 |
| `scripts/build.sh` | 文件 | 统一构建入口；检查 `TARGET` 后调用共享构建实现 |
| `scripts/build_vit.sh` | 文件 | 历史命名的底层共享构建实现，保留兼容 |
| `scripts/run_prepared_vit_b16.sh` | 文件 | 使用已准备 checkpoint 启动完整 ViT 两方推理 |
| `scripts/run_vit_b16_e2e.sh` | 文件 | ViT ALICE/BOB、日志、心跳和结果检查的底层 wrapper |
| `scripts/run_bert_e2e.sh` | 文件 | BERT 数据预检及 ALICE/BOB wrapper |
| `scripts/convert_vit_checkpoint.py` | 文件 | 将真实 Hugging Face checkpoint 转为 BOLT V3 格式 |
| `scripts/vit_reference.py` | 文件 | BOLT fixed-point clear reference |
| `scripts/vit_pytorch_reference.py` | 文件 | 单张图片的 PyTorch 浮点 reference |
| `scripts/evaluate_vit_b16_pytorch.py` | 文件 | ImageNet top-1/top-5 浮点评测 |
| `scripts/download_imagenet_val.sh` | 文件 | 断点续传并校验官方 ImageNet validation tar |
| `scripts/prepare_imagenet_val.py` | 文件 | 解包图片并从 devkit 生成正确类别 manifest |
| `SCI/tests/vit_bolt/bolt_vit.cpp` | 文件 | `BOLT_VIT` 两方主程序、初始化、统计和最终 reveal |
| `SCI/tests/vit_bolt/bolt_vit_protocol.cpp` | 文件 | Patch、Attention、MLP、pre-LN、CLS 和分类头 secure graph |
| `SCI/tests/vit_bolt/vit_model.cpp` | 文件 | BOLT V3 checkpoint 严格 loader |

## 8. ViT 输出文件

| 文件 | 功能 |
|---|---|
| `prediction.txt` | 最终 reveal 的类别编号 |
| `alice.log` / `bob.log` | 两方运行日志 |
| `operator_trace.csv` | 各算子的 shape、scale、packing、时间和通信量 |
| `performance.csv` | 端到端、BFV setup、MPC setup 等阶段统计 |
| `backend_breakdown.csv` | HE、MPC、转换三个部分的统计 |
| `run_metadata.json` | 模型、packing、BFV 和 fixed-point 配置 |
| `reference_report.json` | clear/fixed 与 secure 最终结果检查 |

当前真实 ViT-B/16 fixed-point 与 PyTorch 仍存在已记录的不一致，因此成功启动或
产生 `prediction.txt` 不等于完整正确性验收 PASS。详细状态见根目录
`README_VIT_ZH.md`。
