# OURS ViT-B/16 clean fork

This tree was copied directly from the audited BLB ViT-B/16 runtime. The
immutable A0 copy is `../vit_b16`; all head-heterogeneous development belongs
here. The baseline details below still define the cryptographic donor and data
layout and must not be weakened by later ablations.

本目录把已复现的 BLB BERT-base 安全推理执行图迁移到 ViT-B/16，同时保留
`cpu/`、`gpu/` 中原有的 BERT CPU/GPU 源码和可执行目标。ViT 是独立的
`ckks_vit_b16_main` target，不替换 `ckks_bert_large_main`。

## 模型和结构映射

真实 checkpoint 为 Hugging Face `google/vit-base-patch16-224` revision
`10064cc17b69d623a025c9e90e11230cc8ba6967`。转换后的 canonical 文件是
`checkpoints/vit_b16/bolt_model.txt`，SHA-256 为
`c980c11a8478f3ef172fa1121fc87eb02141ad8d8934953ed0323c60fe7d4715`；
manifest 明确要求 12 blocks、768 hidden、12 heads、3072 FFN 和 1000 类。

执行映射如下：

| ViT-B/16 结构 | BLB 路径 |
|---|---|
| 16×16 patch embedding | 原 BSGS2/fused CKKS MatMul |
| CLS 与 position embedding | Alice 模型参数的 MPC share addition |
| pre-norm LayerNorm | 原 BLB MPC square/mean/rsqrt + CKKS affine |
| Q/K/V、QKᵀ | 原 FineGrainFusion、MHP/BSGS CKKS 路径 |
| attention 的 `1/sqrt(64)` | Q 投影后公开 CKKS scale metadata 精确除以 8 |
| Softmax、Softmax×V | 原 BLB Softmax MPC/HE 和 ct-ct 路径 |
| output projection | 原 BSGS2 CKKS MatMul |
| GELU、FFN1、FFN2 | 原 BLB GELU 和 fused MatMul 路径 |
| final LayerNorm | 同一 BLB LayerNorm 路径 |
| CLS 1000 类 head | 原 BSGS2 路径，1000 输出仅布局补零到 1024 |

checkpoint 有 197 个 logical tokens。原 BLB full-slot compatibility packing
需要二次幂行数，因此固定使用 **197 logical → 256 physical**：前 197 行为
CLS+patch tokens，后 59 行为公开零 padding；attention 的 padding key 使用公开
mask 屏蔽。没有 197-token 专用 packing，也没有新增协议。

由该布局直接推导的主要形状为：hidden `256×768`、每头 attention
`256×256`、FFN `256×3072`。原 BSGS blocking 随现有 slot layout 重算：QKV
为 `N=32768, slots=16384, n1/n2=8/4`，output projection 和 patch 的 BSGS2
为 `4/8`，FFN1 为 `8/4`，FFN2 为
`N=16384, slots=8192, n1/n2=2/8`，分类头为 `4/8`。这些都是原公式在
ViT shape 上的结果。

CKKS context 继续使用 SEAL `tc128`；GPU evaluator 为官方 Phantom DEVICE
后端，CPU evaluator 为 SEAL HOST 后端。MPC 使用 SCI `USE_VOLE=1`，双方各
64 线程。安全非线性边界保持 37-bit ring、scale 12，算子内部 weight/bias
scale 沿用 BLB 的原调度。

真实 checkpoint 的 GPU 明文保留原 BLB BSGS/FineGrainFusion slot 布局，但由
compact semantic slots 直接交给官方 Phantom encoder，不再先扩张成 SEAL/RNS
plaintext 后把数百 GB 的表示逐个拷到 GPU。checkpoint loader 已检查有限值、
定点位宽与 scale；ViT 专用 `encode_trusted` 因此可在 compact host slots 上做
保守系数界检查，跳过每个明文一次的 GPU→CPU FFT 系数回读和全 stream 同步。
原 Phantom `encode` 保持原检查，BERT 调用没有改变。

QKV/BSGS2 的独立外层任务默认使用 8 个 Phantom encoder/CUDA streams；每个
packing/operator 结束都有同步边界，随后释放临时 plaintext。可用
`BLB_VIT_GPU_LINEAR_STREAMS=1..16` 调整并发，用
`BLB_VIT_GPU_DIRECT_PLAINTEXT=0` 退回旧的 SEAL plaintext H2D 诊断路径。
该调度不改变真实权重、packing、密文数、rotation/multiply/rescale、通信或协议。

完整 12 层的全部离线 plaintext cache 同时驻留会超过测试机内存。实现只在
当前层调用原 `weights_preprocess_layer`，该层结束即释放；plaintext 内容、
packing、在线执行图、计数和通信均不变。报告分别记录 offline setup、secure
online 和二者之和，不能把离线调度时间混入在线时间。

## 构建与运行

```bash
docker start blb
docker exec blb bash -lc '
cd /workspace/BLB
JOBS=64 ./scripts/build_vit_b16.sh
'
```

CPU（1 Gbps / 0.3 ms RTT）。下面的命令复现此前 CPU 表格中的
`OMP_NUM_THREADS=16`；MPC/VOLE 的 64 个 protocol workers 保持不变：

```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_OMP_THREADS=16 \
BLB_VIT_STOP_AFTER_BLOCK=12 \
BLB_VIT_TIMEOUT_SECONDS=21600 \
./scripts/run_vit_b16.sh cpu \
  outputs/vit_b16_blb/cpu_full_12blocks 63940
'
```

GPU 运行前先用 `nvidia-smi` 选择两张空闲 A100，再把示例中的 `0 1` 换成
实际空闲编号：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader

docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_OMP_THREADS=64 \
BLB_VIT_STOP_AFTER_BLOCK=12 \
BLB_VIT_TIMEOUT_SECONDS=21600 \
./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_full_12blocks 63942 0 1
'
```

性能排查时只跑一个完整 block（执行路径不变）：

```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_STOP_AFTER_BLOCK=1 \
BLB_VIT_TIMEOUT_SECONDS=3600 \
./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_block1_diagnostic 63944 0 1
'
```

## GPU 性能回归排查（2026-08-29）

迁移检查定位到两处 shape-adaptation 回归：QKᵀ 的 256-token 路径没有按
BLB Figure 7 / Appendix C.1 的 MHP/BSGS 调度重算，曾先后落入错误的历史
rotation cutoff 和完整逐对角展开；Softmax division 也曾被合并到单个
`FPMath` worker。修复只恢复同一 BLB 调度：`256×64=16384` slots、每个 head
一个 ciphertext、MHP factor 1、BSGS `baby=16/giant=16`，以及原来的 48 个
完整 row worker。没有新增 packing、近似、协议或安全参数。

在空闲 A100 GPU2/GPU3、双方 64 OpenMP threads、1 Gbps/0.3 ms RTT 下，完整
单 block 成功目录为
`outputs/vit_b16_blb/gpu_block1_bsgs_parallel_20260829/`，双方退出码为 0/0：

| 项目 | 实测 |
|---|---:|
| 单 block secure online | 214.423 s |
| setup/offline | 305.593 s |
| 实际 Alice wall | 9:23.40 |
| QKᵀ Phantom HE | 5.743 s |
| block HE internal counter | 40.463 s |
| conversion internal counter | 32.836 s |
| plaintext H2D | 76.208 s |
| MPC nonlinear wall buckets | 80.041 s |
| 总通信（Alice+Bob，含初始化） | 9.649 GB |
| full-graph rotation / ct-ct / ct-pt / rescale | 11124 / 3072 / 166584 / 6720 |
| full-graph CKKS→MPC / MPC→CKKS / conversion ciphertexts | 204 / 252 / 456 |

其中 QKᵀ 自身固定为 9720 rotations、3072 ct-ct、9876 ct-pt；独立 CPU、
GPU QK 和 Softmax 中间量已与修复前的正确参考路径比较，误差处于 CKKS
随机噪声重复运行范围。最终正确性仍必须以完整 12-block 的 secure logits 对
float/fixed-clear 参考图验证，不能用单 block 外推代替。

论文 BERT GPU artifact 的 `ckks_bert_large_main` 实际只运行一个 block
（`ATTENTION_LAYERS=1`），线性 kernel 又用复用的 all-one plaintext 模拟权重；
因此其约 2.5 分钟数据不是“真实 checkpoint、12 blocks”的直接基准。

## 真实权重 GPU 性能修复（2026-08-30）

旧的完整 12-block GPU 运行把 150k 级真实 CKKS plaintext/层的完整 RNS 表示
从 host 复制到 Phantom，累计 plaintext H2D 为 1695.74 s，secure online 为
2235.93 s（37.27 min），所以明显慢于 CPU。修复后在 A100 GPU1/GPU2、
1 Gbps/0.3 ms RTT 下的正式目录为
`outputs/vit_b16_blb/gpu_direct_qkv_stream8_full12_gpu1_gpu2_20260830/`：

| 项目 | 修复后实测 |
|---|---:|
| 12-block + final head secure end-to-end | 498.104 s（8.302 min） |
| 12 Transformer blocks wall | 490.631 s（8.177 min） |
| setup/offline | 282.695 s（4.712 min） |
| Alice 全进程 wall（含 setup） | 13:10.77 |
| plaintext H2D | 0.000 s |
| Phantom direct plaintext encode | 102.231 s |
| Alice 最大 RSS | 94.75 GB |
| 双方退出码 | 0 / 0 |

双方 secure logits 与 prediction 文件逐字一致。现有参考验证仍报告 secure
top-1 705 与 fixed-clear top-1 248 不一致；旧 CPU/GPU 完整运行也分别得到
240/709 等不同 secure top-1，因此这是尚未解决的既有数值正确性问题，不应把
本次性能修复的 `PASS` 误写成端到端数值验证通过。

脚本会在 loopback 上设置单向 0.15 ms delay 和 1 Gbit/s rate，对应 0.3 ms
RTT；仅清理由该次运行创建的 qdisc。完整运行结束后验证双方输出，并独立执行
float/fixed-clear 12 层参考图：

```bash
docker exec blb bash -lc '
cd /workspace/BLB
./scripts/validate_vit_b16.sh outputs/vit_b16_blb/gpu_full_12blocks
./scripts/validate_vit_b16.sh outputs/vit_b16_blb/cpu_full_12blocks
'
```

每个正式目录保存双方原始日志、资源峰值、qdisc 证据、逐层/逐阶段时间与通信、
HE 计数、conversion 计数、secure logits 和验证报告。
