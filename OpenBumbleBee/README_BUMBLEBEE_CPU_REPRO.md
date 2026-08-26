# OpenBumbleBee CPU Bumblebee 复现

本文只记录 OpenBumbleBee `origin/main` 的 CPU BERT-base 128-token 复现路径。

## 1. 进入容器和代码目录

使用当前运行中的 `openbumblebee` 容器：

```bash
docker start openbumblebee  # 已经运行时可跳过
docker exec -it openbumblebee /bin/bash
cd /workspace/OpenBumbleBee
```

## 2. 编译

```bash
cd /workspace/OpenBumbleBee

# OpenBumbleBee .bazelversion 固定为 6.4.0；不要使用容器中的 bazel 6.5.0。
# CMake 4.x 已移除 msgpack-cpp 的旧兼容模式，foreign_cc 使用官方 CMake 3.27.9。
test -x /tmp/bazel-6.4.0
test -x /tmp/cmake-3.27.9-linux-x86_64/bin/cmake

export BB_BAZEL_ACTION_PATH=/tmp/cmake-3.27.9-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

/tmp/bazel-6.4.0 --batch \
  --output_user_root=/tmp/bazel-openbumblebee-container-640 \
  build -c opt --jobs=128 \
  --spawn_strategy=local \
  --strategy=Genrule=local \
  --strategy=CppCompile=local \
  --action_env=PATH="$BB_BAZEL_ACTION_PATH" \
  //examples/python/utils:nodectl \
  //examples/python/llm/flax_bert
```

容器当前的 `/usr/local/bin/bazel` 包装脚本会追加失效的旧 `--override_repository` 参数，不要调用它。若首次下载 `bazel_skylib` 等依赖超时，先确认容器内的代理变量指向可访问 GitHub 的代理；当前容器已配置为 `http://127.0.0.1:7897`。首次构建若需要下载 Go SDK 等 Bazel 依赖，应允许容器访问依赖源。

## 编译后确认

```bash
test -x bazel-bin/examples/python/utils/nodectl && echo build-ok
test -x bazel-bin/examples/python/llm/flax_bert/flax_bert && echo flax-bert-ok
```

## Runtime Python 依赖（容器首次使用）

容器默认 Python 是 3.10；不要直接复用宿主机按其他 Python ABI 编译的 `jaxlib`。若目录不存在，在容器内执行一次：

```bash
rm -rf /tmp/openbumblebee-pydeps-py310
python -m pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed \
  --target=/tmp/openbumblebee-pydeps-py310 \
  "numpy==1.26.4" "scipy==1.15.3" "ml-dtypes==0.5.1" \
  "jax==0.4.35" "jaxlib==0.4.35" "flax==0.8.5" \
  "datasets==2.19.0" "huggingface-hub==0.23.5" "protobuf==5.27.3"
```

## 3. 启动两方 CPU runtime
在终端一执行：

```bash
cd /workspace/OpenBumbleBee
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/tmp/openbumblebee-pydeps-py310:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles/spulib"
export SPU_BB_SET_IEQUAL_BITS=16
export SPU_BB_ENABLE_MMUL_PACK=1

bazel-bin/examples/python/utils/nodectl \
  --config examples/python/llm/flax_bert/2pc.json up
```

保持终端一运行。在终端二执行 secure 推理：

```bash
cd /workspace/OpenBumbleBee
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/tmp/openbumblebee-pydeps-py310:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles/spulib"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SPU_BB_SET_IEQUAL_BITS=16
export SPU_BB_ENABLE_MMUL_PACK=1

bazel-bin/examples/python/llm/flax_bert/flax_bert \
  --config examples/python/llm/flax_bert/2pc.json \
  --backend cpu
```

运行结束后回到终端一按 `Ctrl-C` 停止节点。

## 4. 正确性诊断

`--fixed-clear` 使用官方 `REF2K` 固定点诊断路径，与生产用的 BUMBLEBEE secure 结果逐层比较：

```bash
bazel-bin/examples/python/llm/flax_bert/flax_bert \
  --config examples/python/llm/flax_bert/2pc.json \
  --backend cpu \
  --correctness --fixed-clear
```

该模式用于正确性诊断；生产性能结果仍以 `BUMBLEBEE` secure 为准。

## 5. 复现配置和参考结果

- 模型：`bert-base-uncased`，`FlaxBertForSequenceClassification`
- tokenizer：`BertTokenizerFast`
- 输入：`max_length=128`，`padding="max_length"`，`truncation=True`
- 配置：`examples/python/llm/flax_bert/2pc.json`、`BUMBLEBEE`、`FM64`、`f=18`、Taylor exp 6 次
- 示例输入：`Bill whistled past the house.`

容器内三次 secure 时间为 `319.04 s`、`309.07 s`（correctness 内 secure）、`316.34 s`，中位数 `316.34 s`；CPU clear 约 `2.03–2.36 s`。两次完整节点日志的通信均约 `6886.54 MiB = 6.725 GiB`。profile 确认 `f_gelu=12`、`f_exp=12`、`f_batch_mmul=36`、`mixed_mmul=128`、`i_equal=128`，未落入 generic/Cheetah fallback。构建阶段可使用代理，runtime 必须取消代理并保留 `NO_PROXY`，否则 gRPC 会把本地 RPC 错误发给代理端口。
