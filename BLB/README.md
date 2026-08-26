# BLB BERT-base 论文复现

本目录是正式复现目录，不依赖 `.paper_repro` 或外部 Git worktree。

## 布局

- `cpu/`：CPU 论文快照，基于作者提交 `25e8d86`。
- `gpu/`：GPU 论文快照，基于作者提交 `d9af284`。
- `build/cpu`、`build/gpu`：从上述正式源码生成的构建目录。
- `checkpoints/`、`dataset/`：官方 MRPC 权重及论文数据入口。
- `scripts/`：容器、构建和双参与方运行脚本。
- `outputs/`：实验日志与结果。

CPU 与 GPU 是作者历史中分叉的两套实现，不能用其中一套源码覆盖另一套；因此将它们作为两个正式后端保存，而不是临时 worktree。

## 启动环境

在第一个终端运行并保持打开：

```bash
cd /home/dm/mind/Vit-Head/BLB
CUDA_ROOT=/usr/local/cuda-12.3 \
BLB_CONTAINER_NAME=blb_vit_head_env \
./scripts/run_official_container.sh
```

容器内固定挂载点仍为 `/workspace/BLB`，这样构建路径不依赖宿主机目录名。

## 构建

在另一个终端运行：

```bash
docker exec --user 1001:1001 --env HOME=/tmp \
  --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/build_paper.sh all
```

构建参数固定为 Release、`NO_REVEAL_OUTPUT=ON`、VOLE-OT、A100 `sm_80`；没有加入论文外算法优化。

## 运行 CPU

```bash
docker exec --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/run_bert_base.sh cpu \
  outputs/paper_bert_base/cpu_run 63700
```

## 运行 GPU

先在宿主机选择两张空闲 GPU：

```bash
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader
```

将下面的 `1 3` 替换为运行时确认空闲的 GPU 编号：

```bash
docker exec --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/run_bert_base.sh gpu \
  outputs/paper_bert_base/gpu_run 63710 1 3
```

脚本固定双方各 64 线程，并在容器 loopback 上设置 1 Gbps、0.3 ms RTT；退出时恢复网络队列。

论文程序测量一个完整 Transformer block。BERT-base 的 12 层延迟和通信量沿用作者口径，将单层结果乘以 12。

## 正式路径验证

验证输出位于 `outputs/formal_layout_validation_20260826/`：

| 后端 | 双方退出码 | ALICE 单层时间 | 12 层折算 | 12 层通信量 |
|---|---:|---:|---:|---:|
| CPU | 0 / 0 | 35.0142 s | 7.003 min | 3.082 GB |
| GPU | 0 / 0 | 12.6517 s | 2.530 min | 3.072 GB |

GPU 与论文的 2.5 min / 3.0 GB 接近。CPU 正式路径验证时宿主机同时运行其他高 CPU 任务；同一 CPU 二进制在低争用测试中为 33.3037 s/层，即 6.661 min。CPU 新二进制 SHA-256 与迁移前低争用测试使用的二进制完全一致，因此该波动不来自代码或目录重构。

GPU 上游代码的 ALICE `ct_transfer_comm` 计数器遗漏一个密文传输分项；3.072 GB 沿用作者日志中的相同协议分项修正。内核 qdisc 同时记录约 291.9 MB wire bytes/层，零丢包、零积压。
