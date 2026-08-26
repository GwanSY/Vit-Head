# 进入
```bash
docker start openbumblebee
docker exec -it openbumblebee /bin/bash
cd /workspace/OpenBumbleBee
```

# 编译
```bash
# OpenBumbleBee 的 .bazelversion 固定为 6.4.0。
# 不要使用 /usr/local/bin/bazel 包装脚本或 Bazel 6.5.0。
test -x /tmp/bazel-6.4.0
test -x /tmp/cmake-3.27.9-linux-x86_64/bin/cmake

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export BB_BAZEL_ACTION_PATH=/tmp/cmake-3.27.9-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

/tmp/bazel-6.4.0 --batch \
  --output_user_root=/tmp/bazel-openbumblebee-container-640 \
  build -c opt --jobs=128 \
  --spawn_strategy=local \
  --strategy=Genrule=local \
  --strategy=CppCompile=local \
  --action_env=PATH="$BB_BAZEL_ACTION_PATH" \
  --verbose_failures \
  //examples/python/utils:nodectl \
  //examples/python/llm/flax_bert
```

# 运行一
```bash
cd /workspace/OpenBumbleBee
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/tmp/openbumblebee-pydeps-py310:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles:$PWD/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles/spulib"
export SPU_BB_SET_IEQUAL_BITS=16
export SPU_BB_ENABLE_MMUL_PACK=1

bazel-bin/examples/python/utils/nodectl \
  --config examples/python/llm/flax_bert/2pc.json up
```

# 运行二
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
export TRANSFORMERS_OFFLINE=1
export SPU_BB_SET_IEQUAL_BITS=16
export SPU_BB_ENABLE_MMUL_PACK=1

bazel-bin/examples/python/llm/flax_bert/flax_bert \
  --config examples/python/llm/flax_bert/2pc.json \
  --backend cpu
```
