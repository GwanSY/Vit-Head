# BLB BERT：100 Mbps、80 ms RTT

下面命令都在**宿主机**执行，ALICE 和 BOB 都运行在同一个
`blb_formal_env_fix` 容器内。请先按 [主 README](../README.md) 完成容器和
`build_blb_formal` 构建，再顺序运行 CPU、GPU；不要同时占用同一端口段。

当前 formal secure 范围只有 BERT layer 0。以下命令用于验证该安全执行路径和测量
分项，不是完整 12 层论文延迟或准确率结果。

## 1. 设置 100 Mbps、80 ms RTT

容器由 `repro/run_official_container.sh` 以 `NET_ADMIN` capability 创建。双方通过
容器内 `127.0.0.1` 通信，因此必须限制该容器网络命名空间的 `lo`：

```bash
sudo docker exec --user root blb_formal_env_fix bash -lc '
tc qdisc replace dev lo root netem limit 10000 delay 40ms rate 100mbit
tc qdisc show dev lo
ping -c 3 127.0.0.1
'
```

`40 ms` 是单方向延迟，往返约为 `80 ms RTT`。`tc` 才是真正的限速；环境变量或
metadata 记录不会自动限制网络。

## 2. CPU / Microsoft SEAL

```bash
sudo docker exec \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --workdir /workspace/BLB \
  blb_formal_env_fix bash -lc '
BACKEND=cpu \
BUILD_DIR=/workspace/BLB/build_blb_formal/sci \
MODEL_PATH=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
OUTPUT=/workspace/BLB/outputs/bert_layer0_cpu_100m_80ms \
PORT=62000 \
RUN_TIMEOUT=0 \
bash repro/run_original_bert.sh
'
```

日志：

```text
BLB/outputs/bert_layer0_cpu_100m_80ms/alice.log
BLB/outputs/bert_layer0_cpu_100m_80ms/bob.log
```

## 3. GPU / PhantomFHE

下面 `GPU_ID=0` 选择第 1 张 GPU；需要第 2 张时改为 `GPU_ID=1`：

```bash
sudo docker exec \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --workdir /workspace/BLB \
  blb_formal_env_fix bash -lc '
BACKEND=gpu \
GPU_ID=0 \
BUILD_DIR=/workspace/BLB/build_blb_formal/sci \
MODEL_PATH=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
OUTPUT=/workspace/BLB/outputs/bert_layer0_gpu_100m_80ms \
PORT=62100 \
RUN_TIMEOUT=0 \
bash repro/run_original_bert.sh
'
```

确认使用 Phantom 和真实 CKKS device plaintext：

```bash
grep -E 'HE_BACKEND mode=gpu|BLB_VIT_BASELINE.*backend=gpu' \
  /home/dm/mind/Vit-Head/BLB.pre_formal_20260826/outputs/bert_layer0_gpu_100m_80ms/alice.log
```

CPU 与 GPU 使用同一个真实 checkpoint；GPU 路径已具备与 host plaintext 对应的
CKKS level/scale device 镜像。当前仍只有 layer 0，12 层循环、pooler 和 classifier
未接通，因此不得把这次运行标成完整 secure BERT 论文复现 PASS。

## 4. 清除限速

无论运行成功、失败还是中断，最后都恢复容器 loopback：

```bash
sudo docker exec --user root blb_formal_env_fix bash -lc '
tc qdisc del dev lo root 2>/dev/null || true
tc qdisc show dev lo
'
```

若 ALICE/BOB 实际运行在宿主机而不是容器内，才改为在宿主机执行：

```bash
sudo tc qdisc replace dev lo root netem limit 10000 delay 40ms rate 100mbit
sudo tc qdisc del dev lo root
```

不要在宿主机设置 `tc`，却把双方进程放到独立的容器网络命名空间中。

## 5. 如何对照论文

论文 WAN3 配置是 `100 Mbps、80 ms RTT`。论文完整 BERT-base、sequence length 128
的目标数据为：

| 后端 | 论文完整端到端时间 | 论文通信量 |
|---|---:|---:|
| CPU | 17.2 min | 3.0 GB |
| GPU | 13.2 min | 3.0 GB |

这些是**论文目标值**，不是本 README 的 layer-0 实测值。只有接通并验收 12 层和
分类头后，才能做同口径端到端比较。当前完整真实权重 fixed-clear 的 MRPC
`acc_and_f1=88.4437%`，论文为 `88.5%`；该准确率结果同样不能标成 secure PASS。
