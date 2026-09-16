# 研究内容二：统一实验参数

编辑同目录的 `research2.json`。所有相对路径均相对于 `blb_base/`，与启动目录无关。也可以复制配置，为每组实验指定独立的 `paths.output_dir`。

## 准确率预算

```json
"accuracy": {
  "budgets_pp": [1, 2, 3],
  "comparison_budget_pp": null
}
```

`budgets_pp` 是相对全 Softmax 的 Top-1 最大下降百分点，支持任意非负且不重复的预算，包括小数或单个预算列表。每个预算都使用同一目标：在校准集准确率约束内选择预测延迟最低的已验证候选。独立 held-out 准确率只用于最终报告。

`comparison_budget_pp=null` 自动选择列表中最大的预算，供旧的随机/精度优先/均衡消融表使用；不限制其他预算的计划导出。无需在修改预算后再改第二处数值。

## 参数位置

| 分组 | 参数 | 含义 |
|---|---|---|
| `paths` | `model`, `model_manifest`, `preprocess_manifest` | 模型、模型说明、图像预处理说明 |
| `paths` | `public_manifest`, `image_root` | 带标签的公开图像清单与图像根目录 |
| `paths` | `allocator_costs`, `supplemental_costs`, `probe_manifest` | 同一运行配置下的实测成本及训练/验证组合清单 |
| `paths` | `output_dir` | 离线实验结果目录 |
| `accuracy` | `budgets_pp`, `comparison_budget_pp` | 准确率预算列表、旧消融表所用预算 |
| `data` | `calibration_count`, `calibration_start` | 选头校准集大小、起始行（从 0 计） |
| `data` | `heldout_count`, `heldout_start` | 独立测试集大小、起始行；`null` 自动接在校准集之后 |
| `search` | `workers`, `batch_size` | 当前 NumPy 明文计算的进程数、每进程批大小 |
| `search` | `max_frontier`, `max_candidates` | DP 前沿上限、完整明文候选验证数量上限 |
| `search` | `logit_weight` | 单头扰动评分中 logits 相对误差权重 |
| `cost_model` | `cost_ridge` | 成本回归正则强度 |
| `cost_model` | `allow_missing_probes` | 是否记录缺失测量；不会将缺失点伪装成实测，也不会绕过当前成本门槛 |
| `comparisons` | `random_seed`, `random_repeats` | 随机对照种子与次数，次数可设置为任意正整数 |
| `comparisons` | `oracle_layer`, `oracle_heads` | 现有小范围代理 DP/穷举对照范围；4–6 个头，不启动安全测试 |
| `secure` | `image`, `binary_override`, `phantom_library_override` | 现有运行镜像、可选替换二进制及 Phantom 库 |
| `secure` | `input`, `assignment`, `output_dir` | 单次安全推理输入、导出的计划 JSON、独立输出目录 |
| `secure` | `group_packing`, `baseline_attention_packing` | Head 分组与 baseline 快速对角复用，独立设置 |
| `secure` | `conversion_mode`, `gpu_direct_plaintext`, `gpu_linear_streams` | 转换模式、GPU 直接明文编码、线性计算流数 |
| `secure` | `omp_threads`, `port`, `timeout_seconds` | 每方 OMP 线程数、基础端口、超时秒数 |
| `secure` | `gpu_id` | 当前任务固定为物理 7 号卡；双方共享该卡，禁止自动选择其他卡 |

模型结构、头数、网络 1 Gbps/0.3 ms 和完整 12 层安全执行沿用当前实现，不作为本轮新增可调参数。改变执行配置后，成本测量和比较 baseline 应使用同样配置。

清单分辨率为 `100/N` 个百分点：1000 张对应 0.1 pp，512 张对应约 0.195 pp。允许自行设置正整数样本数；较小集合仅适合探索，不能说明细粒度预算在总体数据上成立。校准和 held-out 清单仍检查互不重叠。

## 使用

以下命令在 `blb_base/` 执行。默认配置中的数据/成本路径为 `null`，需绑定实际实验资产；不会自动混用历史成本 CSV。镜像默认沿用现有 runner 的 `ours:fast-preserved-20260901`，不是本轮重新验证的 baseline。

查看实际参数，不导入模型、不使用显卡：

```bash
python3 -B Vit-Head/OURS/blb_base/ours_vit_b16/scripts/run_research2.py --show-config
```

填写路径后运行离线选择：

```bash
python3 -B \
  Vit-Head/OURS/blb_base/ours_vit_b16/scripts/run_research2.py \
  --config Vit-Head/OURS/blb_base/ours_vit_b16/experiments/research2.json \
  --stage offline
```

任一阶段加 `--print-command` 可仅查看传递给原入口的命令，不执行。配置覆盖环境中的同名实验参数。实际运行会保存解析后的配置快照。

离线完成后，每个预算均输出，例如：

```text
outputs/research2/experiment_01/
  research2_config.resolved.json
  budget_candidates.csv
  budgets/1.0/head_assignment.json
  budgets/1.0/head_plan.txt
  budgets/2.0/head_assignment.json
  budgets/2.0/head_plan.txt
  budgets/3.0/head_assignment.json
  budgets/3.0/head_plan.txt
```

`budget_candidates.csv` 区分校准与 held-out 降幅、预算是否通过、预测 attention 成本以及计划哈希。`same_plan_as_budget_pp` 标记相同计划，便于只验证一次。这里不会自动批量启动安全推理，也不将预测成本当作端到端时间。

将 `secure.assignment` 指向选中的 `head_assignment.json`，设置唯一的安全运行输出目录，然后执行：

```bash
python3 -B ours_vit_b16/scripts/run_research2.py --config ours_vit_b16/experiments/research2.json --stage secure
```

该阶段直接复用现有两方容器入口，只开放 GPU 7，运行完整 12 层。修改预算不需要重新编译 C++。敏感度缓存绑定模型、预处理、校准清单、评分参数和实现文件；单独修改预算可以复用缓存，修改这些输入则重新计算。

此入口目前接入现有离线算法及成本门槛。此前规划的 GPU 明文加速、成本模型重构和局部改进搜索尚未在这次参数整理中实现。
