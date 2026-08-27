# BLB ViT-B/16 secure baseline

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

真实 checkpoint 的 GPU 明文先按原 BLB BSGS/FineGrainFusion 布局由 SEAL 在
准确 chain/scale 上生成，再通过已有兼容层传入官方 Phantom evaluator；CPU
直接保留 SEAL plaintext。每个原 fused MatMul 结束后只释放该算子的设备明文，
并清理由 Phantom `UINT64_MAX` pool threshold 留下的空闲页面，避免 QKV 的已
释放页面阻塞 FFN1 分配。该处理只修正资源生命周期，不改变 packing、密文数、
rotation/multiply/rescale、通信或协议；没有加入分块 device encoding 等新调度。

完整 12 层的全部离线 plaintext cache 同时驻留会超过测试机内存。实现只在
当前层调用原 `weights_preprocess_layer`，该层结束即释放；plaintext 内容、
packing、在线执行图、计数和通信均不变。报告分别记录 offline setup、secure
online 和二者之和，不能把离线调度时间混入在线时间。

## 构建与运行

```bash
cd /home/dm/mind/Vit-Head/BLB
docker exec --user 1001:1001 --env HOME=/tmp \
  --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/build_vit_b16.sh
```

CPU（1 Gbps / 0.3 ms RTT、双方各 64 线程）：

```bash
docker exec --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/run_vit_b16.sh cpu \
  outputs/vit_b16_blb/cpu_run 63940
```

GPU 运行前先用 `nvidia-smi` 选择两张空闲 A100，再把示例中的 `0 1` 换成
实际空闲编号：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader

docker exec --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_run 63942 0 1
```

性能排查时只跑一个完整 block（执行路径不变）：

```bash
docker exec \
  -e BLB_VIT_STOP_AFTER_BLOCK=1 \
  -e BLB_VIT_TIMEOUT_SECONDS=3600 \
  --workdir /workspace/BLB blb_vit_head_env \
  ./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_block1_diagnostic 63944 0 1
```

## GPU 单 block 性能排查（2026-08-27）

在 A100 GPU3/GPU4、双方各 64 线程、1 Gbps/0.3 ms RTT 下，audited
SEAL-plaintext → Phantom 路径的最终成功目录为
`outputs/vit_b16_blb/guard_gpu_finalisolated_block1_1gbps_0p3ms_20260827/`，
双方退出码为 0/0：

| 项目 | 实测 |
|---|---:|
| 单 block secure wall time | 129.503 s |
| setup/offline（一次性上下文、VOLE、patch、layer-0 packing） | 329.005 s |
| block HE/fused stages | 68.982 s |
| conversion | 36.527 s |
| plaintext H2D | 57.262 s |
| ciphertext H2D / D2H | 0.106 / 0.173 s |
| MPC nonlinear wall buckets | 31.798 s |
| block 通信（Alice+Bob） | 2.936 GB |
| block rotation / ct-ct / ct-pt / rescale | 1968 / 6168 / 150594 / 3009 |
| block CKKS→MPC / MPC→CKKS / conversion ciphertexts | 426 / 696 / 1122 |

本次排查修复了三项迁移退化：conversion worker 从误设的 1 恢复为要求的
64；offline plaintext 按算子实际 chain/scale 一次生成，不再在线
decode/re-encode；每个 fused MatMul 结束后释放 Phantom plaintext pool 空闲页，
QKV 后 GPU 占用可回落，FFN1 不再因累计页面 OOM。布局适配日志均为本地公开
置换（communication=0），没有重复网络 repack，也没有 512-token padding。

129.503 s/层直接外推 12 层为 25.90 min。即使假设 plaintext H2D 完全为零，
按同一实测时间线的下界仍约 14.45 min，因此在保持真实权重、论文宽位
CKKS↔MPC 转换和 GPU smudging 边界的前提下，不能把 10 min 当作该实现的可达
验收值；要继续降低会需要新的 device-weight 调度/分块或削弱安全步骤，均不在
本迁移范围内。完整 12 层运行目前保持暂停。

脚本会在 loopback 上设置单向 0.15 ms delay 和 1 Gbit/s rate，对应 0.3 ms
RTT；仅清理由该次运行创建的 qdisc。完整运行结束后验证双方输出，并独立执行
float/fixed-clear 12 层参考图：

```bash
cd /home/dm/mind/Vit-Head/BLB
bash scripts/validate_vit_b16.sh outputs/vit_b16_blb/gpu_run
bash scripts/validate_vit_b16.sh outputs/vit_b16_blb/cpu_run
```

每个正式目录保存双方原始日志、资源峰值、qdisc 证据、逐层/逐阶段时间与通信、
HE 计数、conversion 计数、secure logits 和验证报告。
