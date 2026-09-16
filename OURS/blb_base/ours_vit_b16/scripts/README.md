# 脚本目录说明

日常实验从 `blb_base/` 使用 `python3 -B scripts/run_vit.py`，默认读取
[`HAP-ViT.json`](../experiments/HAP-ViT.json)。这里保留实际实现与公共工具。

## 主流程：顶层 14 个脚本

| 脚本 | 职责 |
|---|---|
| `run_research2.py` | 配置读取、阶段路由、统一日志；兼容旧入口 |
| `research2_costs.py` | 缺失安全成本补测、已有成本复用及续跑 |
| `collect_attention_assignment_costs.py` | 从真实双方日志提取完整 attention 成本 |
| `offline_allocator_experiment.py` | 成本验证、准确率预算选头、对照及计划导出 |
| `offline_cost_aware_head_allocator.py` | 成本模型、候选组合与搜索公共实现 |
| `make_composition_probe_plan.py` | 生成一个固定成本组合计划 |
| `make_composition_probe_suite.py` | 生成训练/验证探针清单；后续独立成本实验仍可使用 |
| `run_clear_sensitivity.py` | 单独执行离线明文敏感度计算 |
| `run_true_small_oracle.py` | 单独执行小范围真实明文穷举对照 |
| `vit_reference.py` | 明文 ViT 和固定点参考实现，多个入口共同依赖 |
| `preprocess_vit_image.py` | 将图像转换为安全推理输入 |
| `verify_vit_manifest.py` | 模型导出清单验证 |
| `verify_head_reorder.py` | 当前安全启动入口依赖的头重排校验 |
| `summarize_a3_a4_run.py` | 当前 A3/A4 端到端运行汇总 |

## 分类目录

- `legacy/`：28 个历史方案、论文队列汇总和旧报告工具。包括旧贪心选头、P1、quick-method、two-budget 等；不是当前 Research2 主入口。保留以便复查历史实验。
- `diagnostics/`：9 个张量对照、Softmax/V 排查、真实路径审计与 CUDA 调试文件。不会因普通实验自动执行。
- `tests/`：6 个已有测试文件，集中进行维护；包含旧方法公共工具的测试。

移动后已修正共享模块导入、基于 `__file__` 的项目根目录以及 Shell/文档引用。
历史结果和归档中记录的原始路径未改写；它们保留运行时的事实。

删除了顶层 `summarize_blb_vit_baseline.py`：它与
`blb_base/vit_b16/scripts/summarize_blb_vit_baseline.py` 内容完全相同，当前双方启动器本就使用后者，故只保留 baseline 的唯一实现。
主入口与基础分配器虽都包含搜索相关逻辑，但前者负责实验组织、后者提供公共实现，不能作为重复代码删除。

测试入口（不需要启动 GPU 推理）：

```bash
python3 -B -m unittest discover -s ours_vit_b16/scripts/tests -p 'test_*.py'
```

配置支持 `//` 行注释，字符串内的 `//` 会原样保留。配置快照及
`--show-config` 的输出仍是标准 JSON，便于其他程序读取。
