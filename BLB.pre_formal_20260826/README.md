# CPU
```bash
docker exec blb_vit_head_env \
  /workspace/BLB/.paper_repro/run_pair_once.sh cpu \
  /workspace/BLB/.paper_repro/cpu_paper/build2/bin/ckks_bert_large_main \
  /workspace/BLB/outputs/paper_bert_base_repro_20260826/cpu_repeat \
  63700
```

# GPU
```bash
docker exec blb_vit_head_env \
  /workspace/BLB/.paper_repro/run_pair_once.sh gpu \
  /workspace/BLB/.paper_repro/gpu/build2/bin/ckks_bert_large_main \
  /workspace/BLB/outputs/paper_bert_base_repro_20260826/gpu_repeat \
  63710 1 3
```

# 查看结果
```bash
tail -35 BLB/outputs/paper_bert_base_repro_20260826/cpu_run/alice.log
tail -35 BLB/outputs/paper_bert_base_repro_20260826/gpu_run/alice.log
```bash

# 结果

后端	论文	复现	偏差	通信量
CPU	6.1 min	6.772 min	+11.0%	3.082 GB
GPU	2.5 min	2.587 min	+3.5%	3.072 GB