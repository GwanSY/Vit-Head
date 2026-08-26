# BLB 官方 Zenodo artifact 与容器

官方记录：

- V1 源码：Zenodo `15590214`，`CKKS-MPC-main.zip`，36,378,327 bytes，MD5 `a6a84554a55204689ccb563001595c6c`；
- V2 容器：Zenodo `15621724`，`blb.tar.gz`，6,344,993,332 bytes，MD5 `86393729bc5530f43e95d754138d51e7`；
- V3 补丁：Zenodo `15627952`，`0001-more-flexible.patch`，23,757 bytes，MD5 `b5401f8f59b935cefea4d53f2117bdef`。

Zenodo 说明还提供以下镜像：

```text
crpi-hxmgk7l66ku6bx2x.cn-hangzhou.personal.cr.aliyuncs.com/tianshixu/ppdl:blbv1
```

## 下载并校验

所有文件只落到当前 BLB：

```bash
cd /home/dm/mind/MIND_code/Pencil/BLB
bash repro/fetch_official_artifacts.sh \
  2>&1 | tee outputs/zenodo_fetch.log
```

脚本支持断点续传并逐文件验证 Zenodo MD5。若只需要源码和 V3 补丁：

```bash
FETCH_IMAGE=0 bash repro/fetch_official_artifacts.sh
```

当前机器配置的代理 `127.0.0.1:7897` 没有监听，Codex 直连外网也被阻断。请先启动该代理，或在可直连的终端执行：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
bash repro/fetch_official_artifacts.sh
```

本轮实测日志为 `outputs/zenodo_fetch_20260805.log`，当前状态是 `BLOCKED`，没有把空文件或未校验文件视为已下载。

## Docker 权限

当前用户 `dm` 不属于 `docker` 组，访问 `/var/run/docker.sock` 会得到 permission denied。可以选择始终在 Docker 命令前使用 `sudo`，或执行一次：

```bash
sudo usermod -aG docker "$USER"
```

随后**退出当前登录会话并重新登录**，再验证：

```bash
docker info
```

本轮 `docker pull` 的实际权限失败记录在 `outputs/container_pull_20260805.log`；这不是镜像仓库不可达，阿里云 registry 的 HTTPS challenge 已实测可达。

后续通过 `sg docker -c` 刷新组权限后，官方镜像已实际拉取成功：

```text
image id: sha256:f4bb361e2727d00df1ffdcda2d442fac7f7a22d0b21b442e84bbc5db7e542c0f
repo digest: sha256:f696ecc441daeb1e05daaee6b955e12cdc5ea0cbb1f5335dab57f769167d20b4
uncompressed size: 18,176,988,884 bytes
```

容器内 GPU 和当前 BLB 只读挂载均已 PASS。日志为：

- `outputs/container_image_inspect_20260805.log`；
- `outputs/container_gpu_preflight_20260805.log`；
- `outputs/container_mount_preflight_20260805.log`。

注意：`docker` 组等价于授予接近 root 的主机权限，只应对可信用户使用。

## 导入或拉取镜像

使用 Zenodo tarball：

```bash
cd /home/dm/mind/MIND_code/Pencil/BLB
bash repro/load_official_container.sh \
  2>&1 | tee outputs/container_load.log
docker image ls
```

或者直接拉取作者提供的阿里云镜像：

```bash
docker pull \
  crpi-hxmgk7l66ku6bx2x.cn-hangzhou.personal.cr.aliyuncs.com/tianshixu/ppdl:blbv1
```

## GPU 与运行策略

## 最简单的进入和运行方法

只记住下面几步：

```bash
# 1. 让当前终端获得 Docker 权限（每次新开终端通常不用重复）
newgrp docker

# 2. 进入 BLB
cd /home/dm/mind/MIND_code/Pencil/BLB

# 3. 进入官方容器；同名容器已存在时会自动重新进入
CUDA_ROOT=/usr/local/cuda-12.3 \
BLB_CONTAINER_NAME=blb_formal_env_fix \
bash repro/run_official_container.sh
```

看到容器里的提示符后，你已经在容器里了。输入：

```bash
cd /workspace/BLB
pwd
```

应该显示 `/workspace/BLB`。这里就是当前 BLB，不是容器自带的 `/home/CKKS-MPC`。

## 容器内构建和运行模型

在容器里执行：

```bash
USE_HE_GPU=ON \
BUILD_ROOT=/workspace/BLB/build_blb_container_current \
CUDA_ARCH=70 JOBS=2 \
bash repro/build_formal_blb.sh \
  2>&1 | tee outputs/build_real_layer0_20260805.log

bash repro/audit_formal_blb.sh \
  2>&1 | tee outputs/audit_real_layer0_20260805.log

BUILD_DIR=/workspace/BLB/build_blb_container_current/sci \
MODEL_PATH=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
PORT=46000 \
OUTPUT=/workspace/BLB/outputs/container_formal_real_layer0_20260805 \
RUN_TIMEOUT=120m \
bash repro/run_original_bert.sh \
  2>&1 | tee outputs/container_formal_real_layer0_20260805.log

/usr/bin/python3 repro/accept_single_layer.py \
  /workspace/BLB/outputs/container_formal_real_layer0_20260805 \
  2>&1 | tee outputs/accept_real_layer0_20260805.log
```

正式运行命令现在会先验证 checkpoint 和 `oracle_shares/manifest.json`。还没有生成 oracle 份额时会输出：

```text
SINGLE_LAYER_FORMAL status=BLOCKED reason=three_oracle_manifest_verification_failed
```

先生成并验证份额（宿主机或容器内的 `python3` 需要 NumPy；宿主机生成的目录会随 BLB 挂载进入容器）：

```bash
python3 repro/prepare_bert_oracle.py \
  checkpoints/official-blb-mrpc/weights_txt \
  --output checkpoints/official-blb-mrpc/weights_txt/oracle_shares
python3 repro/verify_bert_oracle.py \
  checkpoints/official-blb-mrpc/weights_txt \
  checkpoints/official-blb-mrpc/weights_txt/oracle_shares
```

然后重新运行两方命令。份额验证 PASS 仍不代表复现成功；必须看到五个实际 `ORACLE_STAGE`，且验收脚本检查 `B_C_MSE < 1e-11`。命令结束后输入 `exit` 离开容器。

本轮容器内实际构建和运行日志分别为 `outputs/container_gpu_build_20260805.log` 和 `outputs/container_formal_run_20260805_v4.log`。

最新单层命令的实际日志为：

- `outputs/audit_real_layer0_20260805.log`：静态审计 PASS；
- `outputs/container_formal_real_layer0_20260805.log`：历史版本两方在联网前因 `three_oracle_not_implemented` BLOCKED；
- `outputs/accept_real_layer0_20260805.log`：严格验收因缺少 `fgf1_ln_attention` oracle BLOCKED。

宿主机 CUDA 目录 `/usr/local/cuda-12.3` 会以只读方式挂载，这是系统 CUDA 依赖，不是其他工程源码或构建产物。
启动脚本会自动寻找宿主机的 `nvcc`，并设置 `CUDA_HOME`、`CUDA_PATH`、
`CUDAToolkit_ROOT` 和 `PATH`。如果自动寻找失败，在宿主机指定实际目录后再进入容器：

```bash
CUDA_ROOT=/usr/local/cuda-12.3 bash repro/run_official_container.sh
```

不要在没有这些变量的情况下直接运行 `cmake`；正式构建脚本会明确传入
`-DCMAKE_CUDA_COMPILER=<CUDA_ROOT>/bin/nvcc` 和 `-DCUDAToolkit_ROOT=<CUDA_ROOT>`。
如果旧容器已经报 `Failed to find nvcc`，先输入 `exit`，再从宿主机重新执行上面的进入命令；
容器镜像本身没有 `nvcc`，单纯 `export` 变量无法补出编译器。
启动脚本还会临时创建与宿主机 UID/GID 对应的容器用户，避免出现 `I have no name!`，
并保持 BLB 构建文件的宿主机所有权。
如果当前窗口已经进入旧容器并显示 `I have no name!@...`，先输入 `exit`，
再重新执行进入命令；旧容器运行时不能使用相同的容器名。

宿主机沙箱外检测到 8 张 NVIDIA A100 80GB，驱动版本 `560.35.03`。取得 Docker 权限后先验证 NVIDIA runtime：

宿主机检查日志为 `outputs/gpu_preflight_20260805.log`。

```bash
docker run --rm --gpus all \
  crpi-hxmgk7l66ku6bx2x.cn-hangzhou.personal.cr.aliyuncs.com/tianshixu/ppdl:blbv1 \
  nvidia-smi
```

为了遵守“只修改和构建当前 BLB”，正式运行不得使用容器内 `/home/CKKS-MPC` 的源码、库或构建产物。下面的脚本只把容器当作工具链环境，将当前 BLB 挂载到 `/workspace/BLB` 并把它设为工作目录：

```bash
cd /home/dm/mind/MIND_code/Pencil/BLB
PULL_IMAGE=0 bash repro/run_official_container.sh
```

进入容器后确认：

```bash
pwd
readlink -f /workspace/BLB
```

然后仍在 `/workspace/BLB` 内执行本仓库 README 的构建和两方运行命令。容器内 `/home/CKKS-MPC` 只能用于只读审计，不得作为正式源码或链接依赖。

## 官方 checkpoint 导入结果

镜像内包含 MRPC、RTE、SST-2、STS-B 四套真实 BERT-base 权重。本轮选择与当前二分类设置匹配的 MRPC，仅将模型数据和输入复制到：

```text
/home/dm/mind/MIND_code/Pencil/BLB/checkpoints/official-blb-mrpc/weights_txt
```

未复制或引用容器内源码、库、脚本和构建产物。导入目录约 403 MB、1017 个文件，包含 12 层 encoder、pooler、classifier、输入、mask 和 labels。manifest 命令：

```bash
/usr/bin/python3 repro/manifest_official_blb_checkpoint.py \
  checkpoints/official-blb-mrpc/weights_txt
/usr/bin/python3 repro/verify_bert_manifest.py \
  checkpoints/official-blb-mrpc/weights_txt --layer 0
```

layer 0 和 layer 11 均已实测 `MANIFEST status=PASS tensors=16 model_layers=12 formal_ready=no`。`formal_ready=no` 是因为五块安全 oracle 尚未在容器中实测通过、且正式 pooling/classifier 路径未接通，并非 checkpoint 缺失；真实 layer-0 输入和 oracle manifest 运行时门禁已经接入 secure main。

容器源目录与 BLB 目标目录的 1017 个文件又做了逐文件 SHA256 列表对比，结果为 `OFFICIAL_CHECKPOINT_COPY status=PASS files=1017 byte_identity=sha256`，日志见 `outputs/official_checkpoint_copy_verify_20260805.log`。

## 容器构建限制

镜像本身没有 `nvcc`、`/usr/local/cuda*` 或已安装的 `libPhantom.so`。容器内 CMake 3.28.4、GCC 9.4 和 SEAL 可用；不挂载 CUDA 时的 host 尝试会因 Phantom CUDA declarations 失败：

```text
Bundled Phantom declarations require CUDA C++ headers even for the SEAL host build
```

实际日志为 `outputs/container_toolchain_20260805.log`、`outputs/container_dependency_inventory_20260805.log` 和 `outputs/container_build_20260805.log`。历史容器构建目录 `build_blb_container_fix` 已经验证过工具链；当前源码必须重新构建到 `build_blb_container_current/sci`，构建脚本会记录 GLIBC 版本，运行脚本会拒绝宿主机产物。容器仍依赖只读挂载的宿主机 CUDA toolkit；它不是完全自带 CUDA 的独立镜像。
