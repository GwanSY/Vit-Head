# OpenBumbleBee PhantomFHE CUDA 后端

本后端只把 OpenBumbleBee 已有的同态求值搬到 PhantomFHE CUDA。默认后端为
`gpu`；只有显式指定 `--backend cpu` 时才运行原始 SEAL 路径。GPU、CUDA 或
PhantomFHE 不可用时会直接报错，不会静默回退 CPU。

## 保持不变的范围

- BFV 与 `scale=1` 精确 RLWE/CKKS-form 系数语义；
- 原始模数、degree、安全参数、packing、转换、模型计算图和通信协议；
- Bumblebee Softmax、GELU、LayerNorm；
- MPC、Ferret、VOLE/OT 仍在 CPU 上执行。

实现没有引入 BLB 的近似 CKKS、FineGrainFusion、MHP、BSGS、融合 MatMul
或其他矩阵算法。设备驻留只覆盖原计算图中单个既有的逐项乘加序列和 packing
蝶形，运算顺序及 Galois 元素不变。

## 依赖与编译

容器内需提供 CUDA 和预编译 PhantomFHE：

```text
/opt/openbumblebee-phantom/include
/opt/openbumblebee-phantom/lib/libPhantom.so
```

GPU 构建：

```bash
/usr/local/bin/bazel-6.5.0 --batch build -c opt --config=gpu \
  //examples/python/utils:nodectl \
  //examples/python/llm/flax_bert \
  //libspu/mpc/bumblebee/rlwe:phantom_evaluator_test \
  //libspu/mpc/bumblebee/rlwe:packlwes_test
```

CPU 构建不依赖 CUDA target，并保留原 SEAL 实现：

```bash
/usr/local/bin/bazel-6.5.0 --batch build -c opt \
  //examples/python/utils:nodectl \
  //examples/python/llm/flax_bert
```

## 运行

默认选择空闲显存最多的 GPU，并运行 BERT-base、128 tokens：

```bash
examples/python/llm/flax_bert/run_openbumblebee_2pc.sh
```

显式选择卡或切回 CPU：

```bash
CUDA_VISIBLE_DEVICES=7 \
  examples/python/llm/flax_bert/run_openbumblebee_2pc.sh --backend gpu

examples/python/llm/flax_bert/run_openbumblebee_2pc.sh --backend cpu
```

模拟同机两方的 1 Gbps、约 0.3 ms RTT：

```bash
examples/python/llm/flax_bert/run_openbumblebee_2pc.sh \
  --backend gpu --shape-network
```

`--shape-network` 需要容器具有 `NET_ADMIN`。脚本在 loopback 上配置
`delay 0.15ms rate 1gbit`，对应约 0.3 ms 往返时延。

## 正确性与指标

逐系数 GPU 对照测试：

```bash
SPU_BUMBLEBEE_HE_BACKEND=gpu SPU_BUMBLEBEE_VERIFY_GPU=1 \
  bazel-bin/libspu/mpc/bumblebee/rlwe/phantom_evaluator_test

SPU_BUMBLEBEE_HE_BACKEND=gpu SPU_BUMBLEBEE_VERIFY_GPU=1 \
  bazel-bin/libspu/mpc/bumblebee/rlwe/packlwes_test
```

`SPU_BUMBLEBEE_VERIFY_GPU=1` 让每次 GPU 求值同时执行 SEAL 参考并逐多项式、
逐模数、逐系数比较；仅用于验收，不应加入性能测试。

每个 BUMBLEBEE executable 结束时输出一行 `[BumbleBeeBackend]`，包含：

- `HE_seconds` / `HE_calls`；
- `MPC_seconds`；
- `conversion_seconds` / `conversion_calls`；
- `H2D_seconds` / `H2D_calls` / `H2D_bytes`；
- `D2H_seconds` / `D2H_calls` / `D2H_bytes`；
- `e2e_seconds`、协议发送调用数及收发字节数。

## A100 实测

以下是 2026-08-25 在容器内完成的实测，不是由 CPU 时间推算的结果。

- GPU：物理卡 7，`NVIDIA A100 80GB PCIe`；在隔离容器内显示为 device 0；
- 模型：`bert-base-uncased`，12 层、hidden size 768、128 tokens，FM64；
- 网络：同容器 2PC loopback，`netem delay 0.15ms rate 1gbit`，目标 RTT
  约 0.3 ms；整次运行 0 丢包；
- 性能运行没有设置 `SPU_BUMBLEBEE_VERIFY_GPU`；逐系数验证另行运行；
- GPU driver 实测 SPU 端到端为 `2467.732474 s = 41.128875 min`；
- GPU 输出 logits 为 `[[0.27584457, 0.24460983]]`；
- 每个 rank 的应用层双向载荷为 `7,071,404,101 bytes = 7.071404 GB
  = 6.585758 GiB`；qdisc 统计的整形链路总发送量为 `8.074045 GB =
  7.519540 GiB`。

GPU 原始分类计时如下。`e2e_seconds` 是每个 rank 的壁钟时间；其余分类时间
是各调用壁钟时间在所有并发线程上的累计值，类别之间允许嵌套，不能相加得到
端到端时间。`MPC_seconds` 来自既有 profiler 的 rank 0 汇总，rank 1 没有该汇总
时记为 0。

| rank | HE s / calls | MPC s | conversion s / calls | e2e s |
|---|---:|---:|---:|---:|
| 0 | 1110.841732 / 4,232,159 | 2454.194193 | 60.806414 / 708 | 2458.223183 |
| 1 | 497.288528 / 1,154,466 | 0 | 59.955339 / 634 | 2445.542234 |

| rank | H2D s / calls / bytes | D2H s / calls / bytes |
|---|---:|---:|
| 0 | 166001.027917 / 6,006,124 / 1,562,104,954,880 | 75828.326201 / 2,367,424 / 498,254,413,824 |
| 1 | 58586.178998 / 1,656,078 / 417,307,033,600 | 29024.969739 / 707,562 / 149,740,584,960 |

H2D/D2H 字节数是每次 SEAL/Phantom bridge 传输的累计量，不是显存驻留量或
网络通信量。

同一份代码、同一模型输入和同一网络整形下，另行实测原始 CPU/SEAL 路径，
driver SPU 端到端为 `343.383527 s`，输出 logits 为
`[[0.27592850, 0.24438477]]`；两次独立协议运行的 decoded logits 最大绝对差为
`0.00022506`。HE 逐系数等价性由上面的确定性算子测试单独验证。该 CPU 数据只
用于后端等价性和协议对照，没有用于推算 GPU 时间：

| CPU rank | HE s / calls | MPC s | conversion s / calls | H2D / D2H s | e2e s |
|---|---:|---:|---:|---:|---:|
| 0 | 3086.499046 / 3,343,528 | 330.825652 | 21.714522 / 708 | 0 / 0 | 334.606680 |
| 1 | 1891.858968 / 1,358,218 | 0 | 19.950598 / 634 | 0 / 0 | 321.862843 |

| backend/rank | protocol send calls | send bytes | recv bytes | 双向 bytes |
|---|---:|---:|---:|---:|
| GPU/0 | 152,753 | 4,001,261,123 | 3,070,142,978 | 7,071,404,101 |
| CPU/0 | 152,753 | 4,001,263,212 | 3,070,134,283 | 7,071,397,495 |
| GPU/1 | 83,095 | 3,070,142,978 | 4,001,261,123 | 7,071,404,101 |
| CPU/1 | 83,095 | 3,070,134,283 | 4,001,263,212 | 7,071,397,495 |

协议发送调用数逐 rank 完全一致。双向应用载荷相差 `6,606 bytes`，即
`0.000093%`；整形链路统计相差 `0.009940%`。这确认后端迁移没有改变协议
调用拓扑或通信规模。

使用 `SPU_BUMBLEBEE_VERIFY_GPU=1` 的测试已通过 BFV 运算、精确
CKKS-form NTT/mod-switch/乘加序列、ciphertext add/sub/negate，以及 degree
1024/2048/4096/8192 的 packing/Galois 逐多项式、逐模数、逐系数 SEAL 对照。
隐藏全部 CUDA 设备后，显式 GPU 和省略 backend 的默认运行都输出
`OPENBUMBLEBEE_HE_BACKEND ... no CUDA-capable device` 并以 134 退出，没有回退
CPU。

与给定的 BLB 论文 Bumblebee GPU 对照值约 `4.3 min / 5.8 GB` 相比，本次
实测时间慢 `9.565x`（多 `36.829 min`），应用层通信量按十进制 GB 计高
`21.9%`。因此当前实现满足精确迁移与后端/协议验收，但**没有达到论文性能
基准**；以上 GPU 数字均为 A100 实测值。
