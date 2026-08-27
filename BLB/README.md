# BERT-base

## 编译 BERT CPU 和 GPU
```bash
docker start blb
docker exec blb bash -lc '
cd /workspace/BLB
JOBS=64 ./scripts/build_paper.sh all
'
```
### 仅编译cpu
```bash
docker start blb
docker exec blb bash -lc '
cd /workspace/BLB
JOBS=64 ./scripts/build_paper.sh cpu
'
```

## 检查二进制：
```bash
docker exec blb bash -lc '
ls -lh \
  /workspace/BLB/build/cpu/bin/ckks_bert_large_main \
  /workspace/BLB/build/gpu/bin/ckks_bert_large_main
'
```

## CPU运行
```bash
docker exec --workdir /workspace/BLB blb \
  ./scripts/run_bert_base.sh cpu \
  outputs/paper_bert_base/cpu_run \
  63700
```

### 实时查看cpu运行
```bash
tail -f /home/dm/mind/Vit-Head/BLB/outputs/paper_bert_base/cpu_run/alice.log
```

## GPU 运行
```bash
docker exec --workdir /workspace/BLB blb \
  ./scripts/run_bert_base.sh gpu \
  outputs/paper_bert_base/gpu_run \
  63710 3 4
```

## 实时查看gpu运行
```bash
tail -f /home/dm/mind/Vit-Head/BLB/outputs/paper_bert_base/gpu_run/alice.log
```

## 提取时间和通信指标
### cpu
```bash
rg -n \
  'total time:|ct_transfer_time:|conversion_time:|other_nonlinear_time:|he_compute_time:|total comm:|ct_transfer_comm:|conversion_comm:|other_nonlinear_comm:' \
  /home/dm/mind/Vit-Head/BLB/outputs/paper_bert_base/cpu_run/alice.log
```
### gpu
```bash
rg -n \
  'total time:|ct_transfer_time:|conversion_time:|other_nonlinear_time:|he_compute_time:|total comm:|ct_transfer_comm:|conversion_comm:|other_nonlinear_comm:' \
  /home/dm/mind/Vit-Head/BLB/outputs/paper_bert_base/gpu_run/alice.log
```


# 正式路径验证

验证输出位于 `outputs/formal_layout_validation_20260826/`：

| 后端 | 双方退出码 | ALICE 单层时间 | 12 层折算 | 12 层通信量 |
|---|---:|---:|---:|---:|
| CPU | 0 / 0 | 35.0142 s | 7.003 min | 3.082 GB |
| GPU | 0 / 0 | 12.6517 s | 2.530 min | 3.072 GB |


# ViT-B/16 

## 进入容器
```bash
docker exec -it blb bash
```

## 编译 blb-vit

```bash
docker exec blb bash -lc '
cd /workspace/BLB
JOBS=64 ./scripts/build_vit_b16.sh
'
```

## 运行完整CPU
```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_STOP_AFTER_BLOCK=12 \
BLB_VIT_TIMEOUT_SECONDS=21600 \
./scripts/run_vit_b16.sh cpu \
  outputs/vit_b16_blb/cpu_full_12blocks \
  63940
'
```

## 跑CPU单层

```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_STOP_AFTER_BLOCK=1 \
BLB_VIT_TIMEOUT_SECONDS=3600 \
./scripts/run_vit_b16.sh cpu \
  outputs/vit_b16_blb/cpu_block1_smoke \
  63940
'
```

## CPU结果
```bash
cd /home/dm/mind/Vit-Head/BLB

bash scripts/validate_vit_b16.sh \
  outputs/vit_b16_blb/cpu_full_12blocks
```

## 运行 GPU

先在宿主机选择两张空闲 GPU：
```bash
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader
```
## 完整GPU
```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_STOP_AFTER_BLOCK=12 \
BLB_VIT_TIMEOUT_SECONDS=21600 \
./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_full_12blocks \
  63942 3 4
'
```

## 跑GPU单层
```bash
docker exec blb bash -lc '
cd /workspace/BLB
BLB_VIT_STOP_AFTER_BLOCK=1 \
BLB_VIT_TIMEOUT_SECONDS=3600 \
./scripts/run_vit_b16.sh gpu \
  outputs/vit_b16_blb/gpu_block1_smoke \
  63944 3 4
'
```

## GPU结果
```bash
cd /home/dm/mind/Vit-Head/BLB

bash scripts/validate_vit_b16.sh \
  outputs/vit_b16_blb/gpu_full_12blocks
```