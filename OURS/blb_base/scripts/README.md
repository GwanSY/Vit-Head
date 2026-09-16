# ViT 运行入口

在 `blb_base/` 下运行：

```bash
python3 -B scripts/run_vit.py                         # 完整 12 层安全推理
python3 -B scripts/run_vit.py --stage offline         # 离线选头
python3 -B scripts/run_vit.py --stage costs           # 续跑缺失成本测量
python3 -B scripts/run_vit.py --show-config           # 查看解析后的参数
python3 -B scripts/run_vit.py --print-command         # 查看命令，不运行
```

默认读取 [HAP-ViT.json](../ours_vit_b16/experiments/HAP-ViT.json)，可用
`--config /path/to/config.json` 指定其他实验。相对文件路径以 `blb_base/` 为基准。
所有阶段的详细日志追加到 `paths.output_dir/run.log`，配置快照位于同一目录。

常用端到端参数集中在 `secure`：

| 参数 | 用法 |
|---|---|
| `backend` | `gpu` 或 `cpu`；CPU 不开放任何 GPU |
| `gpu_id` | 物理显卡编号，默认 7；双方共享这一张卡 |
| `ablation` | 默认 `a0` 全 Softmax；也可选 `a2`、`a3`、`a4` |
| `head_plan` | 固定头组合文件；A0 可留空 |
| `assignment` | 离线导出的 assignment JSON；非空时优先使用其中的 ablation/head_plan |
| `input` | 单张已预处理输入；这是一张图的端到端测量 |
| `image`, `binary_override`, `phantom_library_override` | 沿用当前镜像、二进制和库 |
| `omp_threads`, `port`, `timeout_seconds` | 每方线程数、基础端口和超时 |
| `network_mode` | `netem` 为现有 1 Gbps/0.3 ms；`none` 关闭整形 |
| `min_free_memory_mib`, `max_gpu_utilization_percent` | 显卡启动条件，默认 70000 MiB / 5% |
| `group_packing`, `baseline_attention_packing`, `conversion_mode` | 原有计算路径开关 |

完整说明见 [RESEARCH2.md](../ours_vit_b16/experiments/RESEARCH2.md)。
更换设备或执行模式后应使用对应的独立成本表。当前 `costs` 采集器要求
GPU + netem；CPU 可用于端到端 `secure` 阶段。

## 整理后的职责

| 文件 | 保留原因 |
|---|---|
| `run_vit.py` | 用户统一入口，读取 JSON |
| `run_ours_a3_a4_container.sh` | 共用容器启动实现，支持 A0/A2/A3/A4、GPU/CPU；保留名称兼容现有调用 |
| `run_ours_vit_b16.sh` | 容器内双方进程、计时和日志实现 |
| `run_vit_b16.sh`, `validate_vit_b16.sh` | 独立 donor baseline 及验证，不能当作 OURS 重复代码删除 |
| `build_vit_b16.sh` | 本地编译 |
| `build_ours_container_artifacts.sh` | 在参考容器编译产物 |
| `build_ours_image.sh` | 打包和校验镜像；与编译不是同一职责 |
| `build_paper.sh`, `run_official_container.sh` | 原始复现实验构建和环境入口 |
| `legacy/` | 旧论文批跑、诊断和 BERT 入口，不读取 HAP-ViT.json |

删除的两个入口已合并到 `run_vit.py --stage costs`：

- `run_composition_probe_suite.sh`：重复的探针批测入口，由统一配置、日志及续跑逻辑替代。
- `run_full12_cost_anchor_suite.sh`：硬编码旧日期、旧输出目录，且不能续跑；逐层锚点已由同一 costs 阶段生成和采集。

11 个仍有独立用途的历史脚本移入 `legacy/`，更新了脚本内根目录和互相调用路径。
保留它们便于查阅旧实验，不再与当前入口混放。底层 Shell 与 legacy 脚本仍采用
命令行/环境变量接口；要让 JSON 生效，请使用 `run_vit.py` 或兼容入口 `run_research2.py`。
