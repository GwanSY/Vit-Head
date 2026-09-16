# 历史实验入口

这些脚本保留旧实验语义，不读取 `HAP-ViT.json`。当前实验使用
`../run_vit.py`，见 [统一入口](../README.md)。

- `run_ours_a0_container.sh`：A0 中间张量追踪，区别于常规完整 baseline。
- `run_ours_a2_cost_container.sh`：旧同构 A2 探针。
- `run_ours_full12_suite.sh`：旧论文 CSV 队列和 cohort audit。
- `run_ours_secure_accuracy.sh`：逐图安全准确率批跑。
- `run_ours_p1_secure_comparison.sh`：旧 P1 方法对照，调用逐图入口。
- `run_ours_packing_ablation.sh`：Packed/Unpacked 成对消融。
- `run_ours_three_path_curve.sh`：旧三路径曲线实验。
- `run_two_budget_fixed_cost_candidates.sh`、`run_two_budget_variable_full12.sh`：旧双预算候选批跑。
- `run_bert_base.sh`、`run_pair_once.sh`：BERT 复现及其双方启动器。

其中一些入口要求两张物理显卡，或依赖历史产物。它们不适用于当前仅用 7 号卡的实验；
移动不代表已按新配置重新验证这些历史流程。
