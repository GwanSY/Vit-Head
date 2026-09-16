# OURS 精简审查（2026-09-09）

本次只做引用与构建检查，没有删除代码或历史结果。范围是当前 ViT-B/16 GPU baseline、异构推理和研究内容二离线选头；不保留其他模型/框架的运行能力时，下面的第一批项目可以退出工作目录。

## 当前应保留的主线

```text
blb_base/
  ours_vit_b16/runtime/              异构 ViT 安全推理
  ours_vit_b16/scripts/              明文参考、选头、成本采集、实验入口
  ours_vit_b16/experiments/           research2.json 与使用说明
  vit_b16/                          可独立构建的 A0 对照
  gpu/SCI/                          SCI、SEAL/Phantom、MPC 等依赖
  scripts/                          构建及两方容器入口
  docker/                           镜像定义
  checkpoints/vit_b16 -> BLB/...     共享模型链接
  inputs/vit_b16 -> BLB/...          共享输入链接
  build/container-gpu/              当前配置引用的程序和 Phantom 库
  outputs/research2/                本轮及后续研究二输入/结果
```

算法依赖为 `run_research2.py -> offline_allocator_experiment.py -> offline_cost_aware_head_allocator.py -> vit_reference.py`。因此两个 allocator 文件目前不能直接删除其中一个；先抽取共同库、迁移调用才可合并 CLI。

安全入口依赖 `run_ours_a3_a4_container.sh -> run_ours_vit_b16.sh`，以及重排检查、汇总脚本。构建在 `gpu/SCI/tests/bert_large_bolt/CMakeLists.txt` 内同时定义 BERT、A0 和 OURS 目标。目录名称包含 BERT，不代表可以整体删除。

## 第一批：对当前 ViT 主线无依赖，可一起移出/删除

| 项目（相对 OURS） | 当前大小 | 原因与边界 |
|---|---:|---|
| `Aramis/`, `Athos/`, `Beacon/`, `EzPC/`, `FSS/`, `OnnxBridge/`, `Porthos/`, `SIRNN/`, `sytorch/` | 合计约 36 MB | 根目录遗留通用框架，不在当前 `blb_base/gpu/SCI` ViT 构建、Python import 或研究二运行入口的依赖链内。它们彼此可能有依赖，应以“弃用旧工程”作为整体操作，不要与 `blb_base/gpu/SCI` 内的同名协议代码混淆。 |
| `checkpoints/official-blb-mrpc/` | 约 419 MB | 旧 MRPC/BERT checkpoint；ViT 使用的是 `blb_base/checkpoints/vit_b16` 指向的 BLB 模型。只有不再复现旧 BERT 时删除。 |
| `bert.py`, `setup_env_and_build.sh`, `install.sh.1`, 根目录 `Dockerfile`, `Dockerfile_AI_Validation` | 小 | 旧工程安装/模型/镜像入口。当前 ViT 使用 `blb_base/scripts/build_*` 与 `blb_base/docker/Dockerfile.ours`。随旧工程一起移出，不替代现有运行镜像。 |
| `clash_2.0.24_linux_amd64.tar.gz`, `start.sh`, `clean.sh` | 约 5 MB | 代理安装包与代理环境设置/清理脚本，不参与 ViT 算法或构建；当前代理服务若独立运行，不应把删除包文件误当停止服务。 |
| 根目录 `artifacts/`, `results/`, `REPRODUCTION_STATUS.md` | 小 | 旧复现记录，不是研究二输出。建议先归档，避免丢失历史实验说明。 |

以上是静态依赖范围内的结论，没有执行删除后重建。预计第一批主要释放约 460 MB；不会显著改变当前推理耗时。

## 第二批：合并入口后退役，不能直接一把删除

| 项目 | 判断 |
|---|---|
| `greedy_head_selector.py` | 历史逐步扰动门槛选择器，不是当前预算 DP；历史报告生成器仍引用它，先移除旧报告入口或更新文档，再退役。 |
| `select_two_budget_plans.py`, `make_two_budget_plans.py`, `make_two_budget_suite.py`, `run_two_budget_*` | 旧固定/可变头数实验。研究二按整模型准确率预算选择，可从默认工作流去掉；旧实验复现仍需整套保留或归档。 |
| `make_p1_*`, `make_quick_method_*`, `run_ours_p1_secure_comparison.sh` 等 | 旧候选/报告/安全批跑入口，不由 `run_research2.py` 调用。用统一配置入口承接所需对照后，再连同相关说明一起归档。 |
| `a4_ciphertext_diagnostic_interposer.cpp`, `a4_mismatch_cuda_gdb.cmd` | A4 故障诊断工具，不参与正常构建或研究二离线选择；可移入 `tools/diagnostics`，不必出现在主线列表。 |
| `run_true_small_oracle.py`, `run_clear_sensitivity.py` | 独立实验功能，不是无效重复代码。建议保留为可选工具；小范围穷举能帮助评价近似搜索，明文敏感度可在成本未齐时独立运行。 |
| `offline_allocator_experiment.py` 和 `offline_cost_aware_head_allocator.py` | 有重复 CLI/策略，但前者大量调用后者的数据结构、校准、DP、计划导出。应拆成公共算法库和单一入口，不能直接删文件。 |
| `vit_reference.py` 中 BFV trace/固定点诊断 | 与当前 FP 明文校准同处一个文件；应先拆出诊断部分，并更新调用，再精简。 |

## 第三批：大头是历史产物，不是算法源码

- `blb_base/outputs` 约 1.3 GB，占当前目录的大部分。应保留被 baseline、成本 CSV 的 `source_run`、校准缓存、计划证据引用的运行；其余失败/中断/重复诊断可以打包归档。不能仅按目录包含 `smoke` 或日期旧就批量删除。
- `blb_base/build` 约 234 MB，其中 `container-gpu` 约 94 MB、`gpu` 约 95 MB、两份 Eigen 配置各约 21 MB。当前研究二引用 `container-gpu`；直接运行脚本默认引用 `gpu`，现阶段二者均有消费者。统一构建入口后可删除不用的一份并按需重建。
- `build/fast-preserved` 约 4.4 MB，是保留的快速二进制，baseline 尚未最终绑定前应保留。
- `__pycache__`、不被使用的构建缓存和临时编译日志可以清理，但节省有限。

## 当前不能删除

- `vit_b16/`：独立 A0 对照，不是冗余复制。
- `ours_vit_b16/runtime/ckks_bert*.cpp` 等：文件名沿袭 BERT，但构成当前 ViT 核心。
- `gpu/SCI/tests/bert_large_bolt/`：仍包含 ViT 的 CMake 入口。应先迁移目标定义再讨论裁剪 BERT 源码。
- `gpu/SCI/extern`, `deps`, `src`：CMake 仍链接/配置这些依赖，不能根据“只用 GPU”删除 MPC、SEAL 或 OT 部分。
- `LICENSE` 及保留依赖的许可证。
- BLB 的共享 ViT checkpoint 和输入目标：删除链接目标会使 OURS 无法运行。

建议实施顺序：先归档第一批旧工程，再收敛研究二入口和公共算法库，随后统一构建目录，最后依据引用记录清理历史 outputs。每一步只做针对性的构建/入口确认；不需要额外扩展模型实验。

## 本次日志与输入整理

研究二新产物统一在 `blb_base/outputs/research2/`，历史 outputs 未迁移。`experiment_01/run.log` 追加记录所有启动命令、有效配置、stdout/stderr 和退出码；`last_run.json` 记录最近状态。`offline/` 保存 CSV/计划/缓存，`secure/run_NNN/` 保留每次两方推理的原始证据。终端仅输出开始/结束两行。

数据使用 50,000 行 ImageNet 验证清单，已检查默认前 2,000 张图存在。预处理来自 checkpoint `manifest.json` 的 `preprocessing`，不是另造预处理参数。

成本输入是从现有 GPU 7 快速运行日志采集的 3 个真实训练点：`0S6R6L`, `6S0R6L`, `6S6R0L`。`2S6R4L` 双方退出成功，但缺失完整镜像/计划来源记录，当前采集器拒绝接纳。当前快速配置还缺 4 个 held-out 点和完整跨层成本锚点，因此配置路径有效不等于选头实验已经可以完成。

`allocator_costs` 与 `supplemental_costs` 暂指同一训练文件；加载器按 `(layer, plan hash, source run)` 去重，不把它算作两套独立测量。来源与缺项见 `outputs/research2/inputs/cost_sources.json`。没有混入旧双卡成本或关闭 baseline packing 的 7 号卡成本。
