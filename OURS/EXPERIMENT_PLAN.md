# 头级异构机制消融实验方案

## 当前轮次约束与执行状态（2026-10-08）

本节及文末“本轮补充约束与论文验收格式”覆盖下方历史方案中冲突的安排。本轮仅运行 B（全 Softmax）与 H（固定头身份、无新打包）的机制消融；不启动 HA/HP/HAP/J/BP 实验、288 个单头敏感性实验、成本感知分配器或新 CKKS–MPC 转换打包，也不训练或蒸馏。下方历史记录保留用于追溯，不是本轮执行队列。公共 BLB–ViT 后端、MHP/BSGS/FineGrainFusion 保持冻结。

机器契约：[experiment_contract.json](blb_base/outputs/heterogeneous_attention_20261008T205143+0900/experiment_contract.json)；预生成的 16 个头计划：[fixed_head_plans.json](blb_base/outputs/heterogeneous_attention_20261008T205143+0900/fixed_head_plans.json)。2026-10-08 复核通过全部文件 SHA-256、8 个单 Block 配置和 8 个全网配置的计数、真实头索引及单 Block 隔离约束。固定 Block 为 0（零基，第一编码层）；原头 R 顺序 0–5，L 顺序 6–11，各取所需数量的前缀，其余为 S。全网低比例配置仅修改第 0 层；每层固定配置使用相同索引规则。不能根据精度或成本结果更换位置。正式安全运行前仍须绑定该 Block 已通过验证的原始日志及实际融合边界，不把其结果推广到其他层。

数据更正（2026-10-09，用户明确授权，覆盖此前训练集限制）：现有 ImageNet-1K 官方 validation 的 50,000 张图片按 seed=20261008 类别分层为每类 2 张校准（2000）、5 张选择（5000）、43 张最终保留（43000）。这只是官方验证集的内部划分，不能称为训练集或新的官方测试集。校准只拟合 α；选择集用于固定计划消融；43000 张本轮不得评估，留待后续最终计划冻结。全部方法在相同选择集重新计算原始 FP32、BLB 与异构准确率。不得因没有 train/ 目录停止，也不下载训练集。

当前轮次资产目录为 `blb_base/outputs/heterogeneous_attention_20261009/`，三份清单及类别/不重叠/路径检查见 `splits/audit.json`，修订契约见 `experiment_contract.json`。原 20261008 契约只作历史记录。本轮固定头身份原样复用此前 16 个计划，保留用户此前要求的低比例替换点；α 尚须经小批量检查及 2000 张正式校准后冻结。所有选择集预测保存逐图片配对结果；最终保留集只做清单完整性检查，不读取图片进行推理。

α 使用历史第 1 节既定公式，对 2000 张校准图片与有效 token 累加后拟合一次，保留符号；分母非正或非有限拒绝。冻结逐层逐头值、量化规则与哈希后再正式实验。ReLU 全零行输出零；Linear 原有符号、参考 scale=12 与原生 α 融入 W_O 的额外 20 位小数均需核实记录，不宣称浮点与安全路径逐位等价。

执行顺序：先冻结资产/数据/头计划与 α，再验证全 S 协议参考，随后依表从低比例到高比例完成所有配置的完整明文 ViT 评估，并执行数值门控与安全重复测量。单 Block Top-1 来自仅改第 0 层的完整网络。区分原始 FP32、同一异构计划的协议明文和真实 CKKS–MPC；安全 logits 对本计划协议明文沿用 relative-L2≤0.05、max-abs≤0.5、Top-1 一致，记录异常值，不能事后放宽。未通过数值门控的配置保留明文负结果和诊断日志，但不得给出有效性能结论。

若全网安全资源不足，预先固定代表点为 144/0/0、143/1/0、143/0/1、142/1/1（四类最低替换比例），不得按实测最快挑选。每个正式安全配置至少 3 次，交错执行 B 与异构配置；冻结图片、硬件 UUID、网络、checkpoint、预处理、二进制/库/镜像、密码及数值参数。逐次保存安全计算、请求墙钟、初始化和每请求准备；通信记录双方发送量之和，GB=10^9 bytes。C2M/M2C 分别保存逻辑边界、批次、密文数、耗时；单 Block 记录实际输入状态、融合区域、必要转换和重排及上下游准备。不能单层外推，也不能用内部可能重叠的耗时桶求和冒充端到端时间。

本轮仅交付文末指定的两张 CSV、一张 PDF 和结果 Markdown，原始日志作为证据附件。所有缺测字段为 `NOT_RUN`（论文显示 —），保留失败原因；均值、样本标准差、每次原始记录、三种输出误差及预测、逐图片正确性变化、头计划/α/样本/配置哈希、原始日志索引随结果保存。0.5/1/2 pp 只作预声明分析区间，不是筛除负结果的条件。历史 79.878%、929.69 s、35.943 GB 不填入新的配对表。没有完整网络真实测量时不生成带推测散点的结果图，也不把空表或空图宣称为完成交付。

## 历史方案与审计记录（冲突处以上述本轮约束为准）

# 异构头分配与打包实验实现计划

本文件是唯一实验矩阵及后续实验约定；运行命令见 [运行说明](README.md)。本轮固定实验范围，不新增训练。下文路径除特别说明外均相对 `blb_base/`；“待实现”不代表当前代码已经支持。

执行边界固定为原生 `EdgeCloudRuntime`：云端持有并预处理模型，端侧持有私有图像；七个版本均走 `agree → prepare_layer → execute_node → release_layer`。版本和头计划进入双方公开契约，契约不一致时拒绝执行。不得另建绕过端云框架的版本入口。

主方案为**离线自适应、在线固定**：模型方仅使用授权训练/校准数据，按模型、准确率预算和公开部署环境生成公开 `head_plan`；端云安全推理不根据当前秘密图片改头或分支。逐输入路由只作为另立的扩展，必须核算安全路由、分支隐藏、消息长度与时间泄漏及其开销，不能沿用本方案速度结论。

四项定义的机器入口为 `ours_vit_b16/scripts/experiment_contract.py`；离线 assignment 与运行 `plan_binding.json` 保存定义及哈希。统一入口拒绝缺少定义、声明动态选头、降低约定安全目标或语义计划被改写的 assignment。直接 V1 计划用于手工方案/探针，仍只执行静态计划；定义声明不能替代实际安全参数及数值验收证据。

## 1. 固定模型、数据和函数

| 项目 | 固定值 |
|---|---|
| 主模型 | `google/vit-base-patch16-224`，revision `10064cc17b69d623a025c9e90e11230cc8ba6967` |
| 权重 | `checkpoints/vit_b16/bolt_model.txt`；SHA-256 `c980c11a8478f3ef172fa1121fc87eb02141ad8d8934953ed0323c60fe7d4715` |
| 结构 | 12 层，12 heads/层，head_dim=64，hidden=768，FFN=3072，1000 类；batch=1 安全推理 |
| Token | 224×224、patch=16；197 逻辑 token，256 物理行；59 行 padding 不参与有效 key 归一化 |
| 预处理 | `manifest.json`：RGB，双线性直接缩放 224×224，像素/255，mean=std=[0.5,0.5,0.5]；不另加 crop/增强 |
| 数据集 | ImageNet-1k / ILSVRC2012 validation；`../../datasets/imagenet2012/val/validation_manifest.csv`，标签 0–999 |
| 校准集 | 清单有效数据行 `[0,1000)`；仅此集合用于 α、敏感度、候选选择和预算判断 |
| 独立测试集 | 行 `[1000,2000)`；1000 张，仅最终报告，不依据结果重选计划 |
| 安全正确性子集 | 独立测试集前 50 张，所有版本同一清单；单列样本数及区间，不声称 0.1 pp 精度 |
| 计时子集 | 独立测试集前 10 张；每版本先对首张预热 1 次，再对每张独立运行 3 次，共 30 个有效观测 |
| 预算 | Top-1 相对 B 下降 `[0.1,0.5,1.0]` 个百分点；主表 0.5 pp，其余用于曲线 |

源清单已核对 50000 行，SHA-256 为 `46f96f648082aeee023893391bf39eccf5f841eb0183d16b5313c941119a0f56`，前 2000 张文件存在、路径无重复、标签范围正确。实现时导出并冻结各子集清单、标签映射和预处理文件，检查校准/测试不重叠。已有 `bob_image.txt` 只用于历史回归，不自动算作正式子集。1000 张的 0.1 pp 是经验分辨率，不能据此宣称总体误差保证；本轮不扩展其他模型/数据集。

令 `Z=QKᵀ/sqrt(64)`，只在有效 key 上定义：

- S：`Softmax(Z)V`，协议名 `softmax_mpc`。
- R：`ReLU(Z)/sum(ReLU(Z)) · V`，全零行输出零；协议名 `relu_norm_mpc`。不改成平方、多项式或额外 ε 版本。
- L：`α[l,h]·ZV`，协议名 `he_linear`；无行归一化。α 仅在校准集拟合：令 `X=ZV,Y=Softmax(Z)V`，每头 `α=Σ〈X,Y〉/Σ||X||²`，对校准样本及有效 token 累加；分母非正或结果非有限时拒绝，不另设默认值。记录拟合/量化方法和原始头 ID；线上不再拟合。

浮点与定点实现以 `ours_vit_b16/scripts/vit_reference.py` 为核对入口。冻结 checkpoint 的 LayerNorm ε=1e-12 及实际定点处理；当前非线性边界为 37-bit ring、scale=12，但 CKKS/转换内部精度另有调度，必须逐项从实际运行元数据记录，不能只写“FP32”。“安全感知”只指安全执行成本：所有版本遵循同一 CKKS `tc128` 安全目标及数值验收标准，不单独降低定点精度或 Softmax/GELU/LayerNorm 近似质量换速度。不同形状/路径可在同一目标下重新计算 CKKS 参数并记录完整链；已有 L 路径额外层级不等于降级。HA→HAP 仍严格固定精度及近似，必要参数变化须先证明同等安全和数值质量，不能混入精度降级收益。Linear 的浮点定义不等于定点调度：参考边界 scale=12，原生 α 融入 W_O 使用额外 20 位小数；必须分别记录，不能声称两者逐位相同。

## 2. 唯一版本矩阵

| 版本 | 头计划 | 分配算法 | 新打包 P | 用途 |
|---|---|---|---|---|
| B | 全 S | 无 | 无；保留 BLB 原生优化 | 正式 BLB–ViT 基线 |
| H | 固定手工计划 | 无 | 无 | 异构机制可行性 |
| HA | 无新 P 成本选出的计划 | 有 | 无 | 分配算法有效性 |
| HP | 与 H 相同 | 无 | 有 | 手工计划上的打包效果 |
| HAP | 与 HA 相同 | 不重新搜索 | 有 | 打包的纯执行收益 |
| J | 使用新 P 成本重新搜索 | 有 | 有 | 联合优化增益 |
| BP | 全 S | 无 | 有 | 同类头布局回归 |

H 固定每层原始 heads 0–3=S、4–7=R、8–11=L，12 层相同；α 只拟合一次，不根据测试结果调整头类型。该手工点若精度不合格，照实报告；不能事后挑成“有效 H”。补充与 HA 每层 S/R/L 数量相同的随机头身份对照，seed=20260905、20 次，区分“数量变化”与“分配有效”。

完整网络优先 B/H/HA/HAP；HP/BP 优先单层与微基准；J 后置。对比 H–B、HA–H 时同时看准确率与成本，不只比时间；HAP/HA、HP/H、BP/B 才是配对打包效应。J 不能替代 HAP/HA。

**HA→HAP 固定 checkpoint、原始头身份、S/R/L 定义、α、精度、网络、输入及计时口径；禁止训练、再拟合 α 或重新选头。** 仅允许证明等价的权重重排：同步置换 Q/K/V 输出块及偏置、W_O 输入块。头身份以 `(layer,original_head)` 为准。

## 3. 代码入口与当前边界

| 代码 | 职责 / 后续工作 |
|---|---|
| `scripts/run_vit.py` → `ours_vit_b16/scripts/run_research2.py` | 当前统一入口、配置解析及验收；已接入 `--version`、来源配对绑定，分离正确性与性能结论 |
| `ours_vit_b16/experiments/experiment.json`、`version_matrix.json` | 默认 B；后者是机器可读唯一矩阵，配置中没有独立打包开关 |
| `ours_vit_b16/runtime/ckks_vit_b16_main.cpp`、`edge_cloud_runtime.*` | 完整 ViT 和端云执行入口；B 已有实现，不重做基线 |
| `runtime/hetero_plan.*`、`linear_ckks.cpp`、`ckks_bert.cpp`（均在 `ours_vit_b16/` 下） | 计划校验、同步权重重排、三类头及分组执行；从这里审计 P 改动范围 |
| `ours_vit_b16/scripts/verify_head_reorder.py`、`vit_reference.py` | 原始/物理头顺序等价性，浮点、定点和安全结果核对 |
| `ours_vit_b16/scripts/offline_cost_aware_head_allocator.py`、`offline_allocator_experiment.py` | α、单头敏感度、层内计数 DP、跨层 Pareto、完整模型校准与预算导出 |
| `ours_vit_b16/scripts/research2_costs.py`、`collect_attention_assignment_costs.py` | 成本探针、真实测量及来源校验；已按固定 GPU 对与版本检查来源，HA/J 成本布局严格隔离 |
| `scripts/run_ours_container.sh`、`run_ours_vit_b16.sh` | 两方启动、固定镜像/设备、网络整形及原始结果 |

迁移已完成：运行时、CLI、配置、assignment、日志和 CSV 统一使用 B/H/HA/HP/HAP/J/BP；旧枚举、旧 CLI/配置及专属脚本已删除，不保留别名。B 复用基线；其余版本共用选择性转换与原生 CKKS Linear，按计划是否存在 L 决定额外 CKKS 链，不再按旧消融编号分叉。全 S 退化计划合法。

P 当前复用并整合已有同协议连续头分组、选择性密文打包；H/HA 逐头，HP/HAP/J/BP 分组，B 保留原生布局。版本决定分组策略，不能单独切开关。迁移完成不等于已证明新打包的数值与性能收益，仍按第 4 节验收。

HP/HAP 必须提供来源 H/HA 的 `config.secure.json` 和同目录 `plan_binding.json`。入口检查语义计划、模型/输入/二进制/库哈希与执行配置，并使用来源不可变镜像 ID；禁止重新搜索。底层 C++/Shell 只执行公开计划，正式配对实验统一从 `run_vit.py` 启动。
现有 B 回归证据如 `outputs/optimization_explore_20260917/full12_gpu5_gpu6_c2m24_move/` 的 `plain_fixed_secure_validation.json`：clear/fixed/secure Top-1 均为 248；这是单输入证据。不能继续沿用旧文档“B 一直未通过”的结论，也不能提升为数据集准确率。另有 c2m16 等失败实验；冻结 B 时必须绑定代码、二进制、精度和对应报告，不能随意取最快日志。

`validate_secure_run()` 已删除历史固定时限门槛，报告版本、数值与执行结果；speedup 由同设备同口径的新测结果计算。`vit_b16/` 仅保留 donor 参考用途，论文矩阵 B 使用 `ours_vit_b16_main --version B`。

分配器统一称为**准确率约束下的高效头配置算法**。单头损失仅作搜索代理；准确率由完整模型在校准集验收，不能相加当作保证。DP/Pareto 剪枝及候选上限使结果只限于保留并验收的候选，不宣称全局最优。小空间 oracle 仅能在明确枚举范围和目标下称最优；12 层 attention 成本之和是预测，不是完整网络端到端实测。

成本目标使用完整 attention 组合的 `max(云端 wall,端侧 wall)`，覆盖该区间 HE、MPC、转换、在线重排和通信等待，不将可能重叠的内部桶相加。审计记录 HE 操作、MPC 调用、转换密文数、轮次、双方时间及混合归一化时间；后者包含 HE/MPC，不称纯 MPC 耗时。同步未单独插入计时，包含在区间 wall 中；历史日志缺项记 NA。云端模型重排记录 `run_head_reorder_ms`，属于整次运行离线开销，重复出现在逐层成本行时不得累加，也不计入在线优化目标。完整端侧和 setup 开销仍需由全网运行验收。

## 4. 实现顺序与验收

1. **冻结资产与版本。** 已有机器可读配置、`version_matrix.json` 和逐运行 `plan_binding.json`；继续补齐总实验 `experiment_lock.json`，绑定第 1 节资产、B 证据、代码/二进制/库/镜像哈希及参数；导出 H。保留现有 V1 计划格式：首行 `OURS_VIT_HEAD_PLAN_V1`，次行 `layers=12,heads=12,logical_tokens=197,physical_tokens=256`，列 `layer,physical_head,original_head,protocol,alpha`，恰好 144 行；每层两个 head 编号列各为 0–11 的排列，S/R 的 α=1。线上只读取公开静态计划，不依私有输入选头。
2. **先验证 H 和无 P 异构路径。** 分别检查 S/R/L 单头、混合层、完整网络；确认 L 真实走 CKKS，未偷偷改成 MPC surrogate 或客户端解密重加密。以现有逐阶段计数证明调用路径。B/H/HA/HAP 共用必要的数值修复。
3. **无 P 成本→HA。** 使用同一冻结后端重新采集完整 attention 组合和逐层锚点，不能用每头耗时线性相加。`research2_costs.py` 已支持正式双卡来源校验；新旧成本 CSV 分开。四个训练/四个独立 held-out 组合来自探针清单，保持不重叠。沿用成本门槛：四个 held-out 实测点齐全、延迟 MAPE≤15%、六对排序至少五对正确。搜索参数先固定 workers=8、batch_size=8、max_frontier=1024、max_candidates=24、logit_weight=0.25、ridge=0.001。只用校准集按预算选最低预测成本候选；独立测试只报告。无可行计划或退化全 S 均如实记录。
4. **冻结 HA→验证整合后的 P→HAP。** assignment 保存不变；另导出物理排列/layout 文件。已加入按原始 ID 排序的 `semantic_plan_sha256`（协议+α）及独立 `physical_plan_sha256`，不要因等价置换误判换头。先做 QKV/W_O 重排证明和浮点/定点 oracle，再测安全数值误差；证明每个 padding/mask/拼接操作保持模型函数。
5. **微基准→全网配对运行。** 单层覆盖全 S、S+R、S+L、R+L、S+R+L，以及奇数头组/尾部槽位；HP/H、HAP/HA、BP/B 成对跑，记录真实密文数、利用率、HE/转换成本。完整网络先跑 B/H/HA/HAP；相同输入按轮次轮换版本顺序，失败/超时单列，不算 speedup。最后才可重新测 P 成本并搜索 J。

HA/HAP（及 H/HP）数值验收：先使用 `verify_head_reorder.py` 的现有 oracle 阈值；另预注册安全 logits 对本版本 fixed-clear 的 relative-L2≤0.05、max-abs≤0.5、Top-1 一致，并在安全子集报告最大误差及通过比例。配对安全输出也按这组阈值检查、Top-1 逐样本一致；不要求随机 CKKS 噪声下逐位相同。浮点/定点的准确率差异单列；阈值不过先定位，不在看到结果后放宽。训练/重新选头不属于等价性修复。

## 5. 统一记录和统计口径

正式配置：两张不同 A100 80GB（记录具体 UUID）、每方 OMP=64、MPC workers=64、Phantom direct plaintext=1、GPU streams=8、1 Gbps/0.3 ms RTT netem。同一配对不更换卡、线程、镜像或运行开关。保存 qdisc 实际证据；共享卡、无整形和 trace 重构运行单列为诊断。

在线主指标为双方 `secure_end_to_end_s` 的较大值，包含 embedding、12 blocks、final LN 和分类头。分别报告 offline/setup、online、全进程 wall；不得把它们混加为同一指标。通信为 Alice+Bob 各自发送字节之和，分 initialization/online/total，不再加接收字节。HE/MPC/转换内部桶可能重叠，不用桶之和替代端到端时间。等价性 trace 关闭后才计时。

新增/统一产物（均写 `outputs/matrix/<experiment_id>/`，不覆盖历史）：

| 产物 | 至少包含 |
|---|---|
| `experiment_lock.json` | 模型/预处理/数据切分哈希、代码提交与 dirty diff 哈希、二进制/Phantom/镜像 ID、种子、搜索参数、预算、精度与 CKKS 完整参数、设备/网络/线程、计时边界 |
| `plans/<version>/` | assignment、V1 物理计划、144 个原始头协议/α、两种计划哈希、重排映射、校准来源；HAP 引用 HA 语义计划 |
| `costs/*.csv` | layer、S/R/L 数量、计划/布局/二进制/网络哈希、实测或预测、成本作用范围、latency、bytes、rounds、source_run；无 P 与 P 分库 |
| `accuracy.csv` | version、预算、计划哈希、split、N、clear/fixed/secure、Top-1/Top-5、相对 B 降幅 pp、置信区间、数值误差；50 张 secure 与1000张 clear 不混用 |
| `runs.csv` | run_id、version、预算、样本 ID/输入哈希、repeat、warmup、两方退出码、正确性/计时资格、两方及 max online 秒、setup/wall、三类通信 bytes、来源目录 |
| `operators.csv` | version/layer/stage、S/R/L 头数、group 数、密文数/有效槽位比例、rotation、ct-ct/ct-pt、relinearize、rescale、C2M/M2C 次数与密文数、rounds、HE/MPC/转换耗时 |
| `comparisons.csv` | 配对版本/预算/语义哈希、有效样本数、延迟均值/标准差/中位数、通信、准确率；B/目标版本与 HA/HAP 分别计算 speedup |

保留每次运行的原始日志、配置、网络证据和校验 JSON。成本预测与端到端实测分列；相同预算选到相同计划要去重并标记，不能当独立结果。若现有程序缺失某个计数，先补采集或记录 NA，不填 0。计时置信区间按样本聚合/配对，不能把同一图片重复三次当三个独立准确率样本。

交付完成条件：B/H/HA/HAP 均有可复跑配置、冻结计划和正确性证据；HA/HAP 语义哈希一致且只差 P；成本、精度与端到端表来源可追溯。HP/BP 单层结果齐全；J 单列可选增强。新报告输出 CSV/JSON，人工说明只维护本文件及 README。

迁移验证：C++ GPU 构建、七版本双角色 CLI 计划校验、端云握手与 Python 回归通过；B 已在 GPU 0/1、正式 LAN 下重跑完整密态推理并通过单图数值验收，H/HA/HAP 全网仍待独立验收。旧源码备份见 `../document_archives/ours_version_migration_20260920.tar.gz`。

性能审计约定：`validation.json` 使用 `new_heterogeneous_packing` 明确表示矩阵中的新 P；B 的 QKV/BSGS/头对密文打包仍存在，不能把旧 `packing=false` 解读为所有打包关闭。当前 QK/AV 为恢复真实 checkpoint 正确性的完整对角线路径，本地 BLB donor 的全 1 占位权重及注释掉的 AV 掩码不可作为性能修复。历史 B 只代表当时的正确性实现；当前严格恢复版本按第7节重新验收，不继承历史冻结结论。

共用 QK/AV 优化 `direct_target_v1` 通过按目标块二进制位选择旋转方向省去额外对齐，保持原有掩码、rescale、数值精度和端云消息；这是各版本共用执行修复，不改变唯一矩阵或定义为新 P。验证须包含异号/异值头对槽位测试、GPU QK/AV oracle 和完整网络。逐层 `block_online_profile_party*.csv` 新增旋转、ct-pt、ct-ct、rescale、转换密文及编码计时，阶段计数之和必须与 block 总计一致。编码计时是 worker 累计时间取最大值的内部指标，不可与 HE/转换桶相加；转换总桶不能称为 signed 专项耗时。审计产物统一为 CSV/JSON。

2026-09-20 B 修复验证：36 项 Python 测试及 GPU 稠密 QKV/QK、AV/W_O oracle 通过；完整网络见 `outputs/matrix/B_rotation_reuse_20260920_r01/`。与同设备/网络/资产的旧 B 单次对比，在线 839.958→802.777 s，旋转 605656→534232，通信仍约 20.91 GB；clear/fixed/secure Top-1 均 248，relative-L2=0.00578084、max-abs=0.0388184。阶段计数与整层总数一致，精度/HE 参数核对一致。该点不是数据集准确率或稳定加速比；完整审计见同目录 `performance_audit/` 与 `outputs/performance_audit_20260920/inspection.json`。

## 6. 模型权重缓存：独立系统实验，暂不改 B

审计结论：`encode_semantic_plaintext_to_device()` 仅依赖云端固定权重槽位、公开 CKKS 参数和 scale，不使用图片、密钥或加密随机数；具备跨请求缓存条件。当前每次进程启动处理一张图，`weights_preprocess_layer()` 生成槽位，QKV/BSGS2 按次编码并释放，`release_preprocessed_layer()` 清掉本层缓存。只改计时分类不会减少这项实际工作。BSGS2 会按实际输入的公开 chain/scale 对齐；离线描述符必须由静态执行计划传播，并在使用前核验，不能只按层号命中。

本次只读审计为 `ours_vit_b16/scripts/audit_weight_cache.py`，产物 `outputs/performance_audit_20260921/weight_cache/`。按真实运行编码数、N 和活跃 RNS limbs，完整 NTT 明文载荷为 `count × N × limbs × 8`：

| 算子 | 12 层编码统计秒 | 单层编码权重 GiB | 12 层编码权重 GiB |
|---|---:|---:|---:|
| QKV | 27.919 | 40.5 | 486 |
| W_O | 7.619 | 9 | 108 |
| FFN1 | 34.651 | 54 | 648 |
| FFN2 | 52.956 | 45 | 540 |

合计权重 1782 GiB，偏置另 1.6875 GiB；不含 embedding、classifier、密钥、激活和分配器开销，是未压缩载荷估算，不是实测峰值。原始打包 double 槽位仅权重约 162 GiB；候选 int64 多项式系数约 324 GiB，但必须证明范围、舍入和编码一致性，不能直接视为已经可用的精确缓存。部分偏置 scale 超过 100 bits，不得统一压成 int64。全网 GPU 常驻不可行，小容量循环 LRU 也可能逐层遍历后全部失效。仅持久化 double 槽位主要减少已有 setup 中的打包，GPU 编码仍会执行，不能据此扣除 124.88 秒。

建议先做 **FFN2 的云端主存完整明文缓存 + 固定分块 GPU 预取**，仅验证一层/若干分块，再决定是否扩到 12 层；不立即分配数百 GiB。缓存命中后省 FFT/RNS/NTT 编码，但每请求 H2D、缓存查找、同步和等待仍算在线。FFN2 全部权重 540 GiB 若每请求搬运，相对 52.956 秒编码桶的算术带宽参考约 10.95 GB/s；这不是实测盈亏点，必须测端到端。显存使用固定额度，先试 1–4 GiB，pinned 主存采用较小窗口，双缓冲配 CUDA event 管理生命周期；避免逐明文 `save/load` 的 pinned 分配/释放。异步拷贝/计算重叠条件参考 [CUDA 12.3 指南](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)。预取调度只由公开层/分块决定。

实施与验收顺序：

1. 先拆分测量编码的准备/分配、H2D、FFT、RNS/NTT及真实关键路径；124.88 秒不是可直接从 802.777 秒扣除的独占时间，重叠桶也不能用于严格瓶颈排序。
2. 缓存键绑定 checkpoint、原始/物理头计划及 α、算子/分块、padding、N、完整模数值、目标层级、scale 的精确位模式、权重量化定义、NTT 布局和编码器/库版本。权重和缓存均留在云端；缓存独立于请求生命周期。密态输入、随机掩码、OT 材料及用户密钥不纳入此复用。仅参数兼容的请求可共享明文缓存。
3. 先验证同一个明文的导出/回载 RNS 数据、level、scale 完全一致及异步消费前事件依赖；再测真实 FFN2 输出和完整网络原阈值。缓存失配回退编码并记录，不修改 scale 迁就缓存。
4. 若完整明文的主存/PCIe 成本过高，再单列“FFT/整数系数中间缓存”，保留在线 RNS/NTT；这是编码器改造，需独立证明，不能当作现成能力。跨请求同层批调度也只作吞吐扩展，单列排队延迟且不混合不同用户的密文密钥。
5. 分别记录模型部署建缓存时间/字节、冷请求、热请求及重复请求的摊销成本；在线保留 H2D/等待，保存命中率、编码调用数、host/GPU 峰值、缓存预算、通信和数值误差。不得只把旧计时桶移到 offline。部署成本按实际请求数摊销，而非假设无限复用。

该方案尚未实现或测速。B 的 802.777 秒记录和二进制不改；用 `execution_profile` 作为独立部署条件，不增加 B/H/HA/HP/HAP/J/BP 之外的版本。若后续采用缓存，各版本使用同一缓存政策/预算，HA→HAP 也冻结该部署条件；冷 B 对热 HAP 的差异不能归因于异构头或 P。

## 7. Strict BLB-ViT 正确性冻结（2026-09-22）

802.777282 s 是旧实现的历史正确性记录，不是 Strict 版本的结果。旧二进制、Phantom 和源码已保存到 `outputs/strict_blb_20260922/before/`。旧 `b_baseline_freeze_20260922` 报告采用较弱归属口径，不能证明 C 类算法变化为零。

B 仍为全部 Softmax、原头顺序、α=1；矩阵及端云所有权不变。严格恢复要求：

| 项目 | 实现与验收 |
|---|---|
| GELU | BLB 算法4系数；HE 求 x²、x³、x⁴及 F0/F1，MPC 选择区间；fixed 同步定义。 |
| Softmax | t=6、T=-13，HE /64、六次平方、最终乘法；SCI rsqrt 后平方得到倒数，保留 signed/广播修复，禁用分母窄环和倒数裁剪；fixed 同步该求值图。 |
| pre-norm Fusion | HE 执行 AV→WO→残差→LN2统计，FFN2→残差→下一层LN1统计；MPC 仅 rsqrt，首层单独实例化 HE 统计图。跳连分享另行保留。 |
| MPC→CKKS | 本地宽整数 Encode 后重新执行系数分享 ring-to-field，逐 RNS 素数生成残值，再客户端加密、云端合并；输入 FFT 扩展/保护位属于数值实现，须单独记录。 |
| 禁用创新 | 不执行异构头、allocator、selective conversion、新 P。 |
| 冻结门槛 | 源码/二进制/运行绑定、上述协议项复核、完整12层、双方输出一致、secure/fixed相对L2≤0.05且maxabs≤0.5、Top-1一致。论文150 s不作门槛。 |

冻结清单为 `ours_vit_b16/experiments/b_baseline_freeze.json`，实际运行/严格审计为 `outputs/strict_blb_20260922/strict_full_r02/`。本轮完整12层及全部严格门槛通过，B 活跃分支 C 类算法变化为0；唯一版本矩阵不变。运行时 `candidate_not_frozen` / verification=0 为构建时标签，事后状态以哈希绑定的严格审计报告为准。论文图10中 Softmax 最终 smul 为未分组算子，Block4 从 AV 开始；保留独立 HE 乘法及原布局转换符合该分块，不额外做 smul→AV 融合。表6的 Block4 是7个60位数据模数，HE2现使用7个数据模数加1个密钥切换模数，N=32768，仍强制 tc128。不得自动刷新旧清单哈希取得 PASS；新编译文件须重新审阅和验收。

Algorithm 2 的逐步对应（详细参数/误差见 `outputs/strict_blb_20260922/algorithm2_restoration.json`）：

1. 输入仍为两方加法分享；37→77 bit 有符号扩展、16 个保护位和宽整数 FFT 解决 ViT 数值范围。各方独立 Encode，保留各自舍入误差，不把“两次局部舍入”替换成“一次重构后舍入”。
2. Encode 后形成系数分享 `c_b mod 2^53`，再安全扩展到 `R=2^93`，恢复算法2的系数域 ring-to-field 次序。以 scale=12 的输入为例，编码 scale=28；其他输入 scale 同步加16。
3. 在附录C.3的正常事件上，`a_cloud+a_edge−R=c_signed`。每个 CKKS 素数 q_i 分别计算 `f_cloud=(a_cloud−R)·2^(s_out−28) mod q_i`、`f_edge=a_edge·2^(s_out−28) mod q_i`，其和恰为同一编码系数的缩放后模 q_i 值。不能把某一素数残值复制到其他 limb。
4. 各 limb 做线性 NTT；端方加密其分享，云方加自己的明文分享，得到该编码重构的 CKKS 密文。此等式针对所用局部 Encode；不宣称与供体64-bit编码器逐bit相同。C.3统计失败项按每个系数计，整网须联合界定，不能把单次2^-40写成整网保证。

QK/AV 的 `direct_target_v1` 只将原分块旋转的公共对齐合并进旋转方向，掩码、头对槽位、完整对角线乘加和协议边界不变；不是新 packing。公共 padding 的删除/补零属于197→256布局适配，不依赖秘密图片或选头。

冻结结果：GPU 2/5、1 Gbps / 0.3 ms RTT，在线1505.218348 s；两方 setup分别300.998125/324.434053 s，setup+online分别1806.205482/1829.652401 s；在线发送量两方合计34,037,567,058 bytes。clear/fixed/secure Top-1均248，双方logits相同，secure/fixed相对L2=0.01733657865、maxabs=0.137451171825。二进制SHA-256为 `18842f7b1708e95a15cf3c21bd750008c10fcff69425004fd0fa3095c287f348`。这是单图数值/执行验收，不是ImageNet准确率或多次延迟均值。

GELU、转换/Softmax微测试、端云握手及完整网络均通过。诊断轮 `strict_full_r01` 虽数值通过，仍含可选窄环优化，不作为正式B；正式轮为 `strict_full_r02`。后续 H/HA/HAP 的成本与精度必须相对此严格后端重新采集，不混用802.777 s历史基线或其成本表。

## 8. B 性能合理性审计与实现修复（2026-09-27）

第7节保留历史正确性/协议冻结证据；1505.218348 s 的性能状态为 **NEEDS_AUDIT**。不能以旧 BERT 的155 s要求定量复现：原记录实际为单个合成 block 云端12.9366 s×12=155.2392 s，端方为14.2319 s；3.072 GB又含补算。此次使用同机 GPU 0/1、64线程、1 Gbps / 0.3 ms重新采集单层指纹，供体云端23.336 s，ViT block为132.621 s（另有patch/final LN/classifier）。计数副本的耗时只作诊断。

| 云端 HE API 计数范围 | BERT128 | ViT256 | 倍率 |
|---|---:|---:|---:|
| 五个融合块旋转 | 3342 | 44628 | 13.35 |
| ct-ct（含平方） | 3294 | 12780 | 3.88 |
| ct-pt | 57187 | 154277 | 2.70 |
| rescale | 4005 | 12164 | 3.04 |
| QK 旋转 | 1536 | 21504 | 14.00 |
| AV 乘法路径旋转 | 864 | 6144 | 7.11 |
| AV 前对角线布局旋转 | 0 | 15360 | 不定义 |

该表不含 ViT 首层独立 LN、patch、末尾 LN和分类头；完整计数及函数路径见 `outputs/shape_audit_20260927/all_api_counts.csv`。API 调用不是底层 CUDA kernel/key-switch 次数。QKV/WO/FFN1 的 BSGS ct-pt 实际4倍：输入密文数和对角分块均翻倍；FFN2为2倍。QK头对输出2→8 ct，总计12→48 ct；对应 CKKS N没有变大，未发现超过4倍的形状阈值跳变。

超过预期的旋转不能直接删掉：供体 `gpu/SCI/tests/bert_large_bolt/linear_ckks.cpp` 的 QK 注释掉特征归约，AV 省略掩码/对角广播且限制旋转；`ckks_bert.cpp` 的 V加密全1、GELU选择输入也是常数1；Softmax行宽计算为32而非128。Strict执行真实输入、完整对角线及论文F0/F1。这些是不同实际工作量，供体计时退出码0不代表模型数值通过。

完整12层转换计数：BERT单层外推 M2C=2304、C2M=1104 ct，Strict实测5784/3725，分别2.51/3.37倍，总计2.79倍。稳态block转换调用 M2C 9→12（真实V、attention残差、FFN残差），C2M 8→12（QK/V按scale分开、两处pre-norm跳连导出、真实pre-GELU输入）；融合主链仍保留。signed宽整数FFT、保护位及逐RNS转换会增加每ct成本；不恢复供体零掩码或错误degree。he1/he2数据层数及FFT精度适配在此次修复前已经存在，参数未再次改变。patch仅执行一次，context仅初始化一次，旧完整运行H2D/D2H合计不足1 s；权重设备编码仍约99.59 s，缓存实验未启用。

**本次唯一运行时代码修复**：`ckks_bert_basic.cpp` 的 exp安全处理原先两段都用SCI通道0，单层26.704+23.824=50.528 s；按公开固定分段使用已有64通道，保留同样的LT/MSB/mux。输出已执行 `C(z)=max(z,0)`，调用方立即再执行一次；按 `C(C(z))=C(z)` 删除重复检查。无中间HE运算、无新精度/近似/转换/packing，环和模型函数不变。转换/指数/倒数测试8通道通过；64通道完整网络的数值与时间单独记录到 `strict_repair_full/native/validation.json`。原冻结产物不覆盖，修复不能自动继承旧哈希的PASS，研究矩阵不变。

修复后完整12层：**932.614689 s**，clear/fixed/secure Top-1均248，双方logits一致，相对L2=0.01730556948、maxabs=0.155517578125，原门槛全部通过。指数两段检查每层合计0.936～2.511 s；HE/转换计数与旧完整运行保持相同。云端非线性时间桶762.703→221.486 s、HE桶385.610→382.648 s、转换桶428.294→367.316 s、权重编码99.594→104.252 s；桶有重叠，不求和归因。在线通信34.03757→33.56085 GB。历史GPU2/5，本次GPU0/1，均为单次；38.04%仅是观测降幅。当前结论为实现退化已修复、协议及单图数值通过，**最终性能基线尚未冻结**。证据见 `repair_protocol_review.json`、`repair_full_comparison.csv`；新二进制SHA为 `a171862980c147c383e8b8ad48814d415f0c3a67a8b09fcc3415929e217f1c58`，其余277份原审阅源码及Phantom哈希均未变。

## Rotation / 通信专项审核（2026-09-27，接续上次修复）

证据：`blb_base/outputs/wire_rotation_audit_20260927/`。同机 GPU0/1、64个 SCI 通道、1 Gbps / 0.3 ms RTT；BERT128跑一个供体测试层，ViT256跑连续两层，首层与稳态层分列。**统计口径一致不等于两边实际模型计算等价**：供体占位输入、省略运算以及既有精度/转换差异均保留并披露，本次不修改任何协议算法或安全参数。

### 旋转：相同五个融合块、云端源码 API 调用

| 算子 | BERT128 | ViT256稳态 | 倍率 |
|---|---:|---:|---:|
| QKV | 144 | 288 | 2.00 |
| QKᵀ | 1536 | 21504 | 14.00 |
| Softmax×V | 864 | 6144 | 7.11 |
| Output Projection | 84 | 168 | 2.00 |
| FFN | 486 | 780 | 1.60 |
| repacking/other | 228 | 15744 | 69.05 |
| 合计 | 3342 | 44628 | 13.35 |

含零步调用12/24次；去除后3330/44604，13.39倍。首层独立 LN 另有144次，不混入五块表。不是 CUDA kernel 或 key-switch 次数，不能与旧手写计数混用。

- **QK**：供体 `gpu/SCI/tests/bert_large_bolt/linear_ckks.cpp:1349` 注释特征归约；ViT `runtime/linear_ckks.cpp:3220` 两个 token 旋转共6144次、`:3261` 特征树15360次。不是漏计已执行路径，而是供体没有执行归约。
- **AV**：供体 `linear_ckks.cpp:2058` 旋转预算72，后续直接复用未旋转输入；`:2233` 的对角广播被注释。ViT真实输入执行6144次值旋转及15360次掩码后广播，后者列入 repacking。repacking其余384次来自 LN 统计；没有把同一调用同时计入 AV 和 repacking。
- **256布局逐项验算**：令头对数6、槽数16384、token数L、每密文特征块数 `b=16384/(2L)`、每头对特征密文数 `c=64/b`。完整当前求值图的 token 旋转为 `2×6×L×c`，特征树/对角广播分别为 `2×6×L×log2(b)`。128→256时 `b:64→32`、`c:1→2`，对应token旋转1536→6144、树9216→15360；完整QK合计10752→21504。128列是公式推导，**不是已修复或实测 BERT**。
- 两层各调用点步长直方图均与上述循环完全一致；QK旋转上限1024覆盖全部迭代，AV按每个值密文执行256个位置，没有触发替代分支。树每级消费更新后的部分和，左右分支处理不同掩码，不是重复旋转同一操作数。`packed_rotation_test.log` 检查正负随机值、两半槽隔离和 stride，PASS。当前未发现重复执行/重复repack/错误stride；上述公式只证明当前完整求值图的计数，不证明其为所有 BLB 实现的最小成本。

### 通信：每层 Alice_sent + Bob_sent（十进制 MB）

| 类别 | BERT128 | ViT256稳态 | 倍率 |
|---|---:|---:|---:|
| CKKS→MPC | 243.665 | 960.772 | 3.94 |
| MPC→CKKS | 7.833 | 940.184 | 120.04 |
| nonlinear MPC | 84.728 | 327.431 | 3.86 |
| ciphertext transfer | 164.038 | 535.292 | 3.26 |
| other | 0 | 0 | — |
| 合计 | 500.263 | 2763.678 | 5.52 |

ViT首层含初始 LN，总计2861.102 MB、5.72倍。每方每层均通过“各项之和=单调发送计数差”的精确字节守恒检查；ViT还与 `block_online_comm_bytes` 逐层核对。`NetIO::send_data` 计数，不计recv、不叠加旧桶；主HE通道及全部SCI io/io_rev/io_GC按指针去重。构造、keygen、已执行的离线OT/VOLE与warm-up在起点前排除；在线协议按需触发的OT扩展/补充仍算在线，不能假定已经离线完成。数值是应用层序列化字节，包含长度头、不含TCP/IP头。M2C/C2M为转换函数内SCI消息，外部密文序列化发送统一归 ciphertext transfer，类别不重叠。

初版探针未包住 `HE_CKKS::encrypt_send`，但该函数已有精确发送日志：首层325871435、稳态303848653 bytes，与残余主通道字节逐字节相等，因此报告归入 ciphertext transfer；不是丢弃 other。原始日志保留，后续探针直接包住此调用。

**120倍的来源与边界**：供体 `ckks_bert_basic.cpp:280` 在本地编码后调用一次 `ring_to_field`；`nonlinear.cpp:454` 使用全零MSB、至多64位扩展及本地取模。ViT `runtime/ckks_bert_basic.cpp:675` 编码前做37→77位安全扩展，整数FFT带16保护位，之后再做53→93位系数扩展，最后本地逐RNS取模。两次扩展分别消费槽分享和FFT系数分享，未发现重复处理同一中间值；RNS循环未重复执行安全扩展。不能将供体已知MSB捷径直接用于这些有符号输入。本次只定位开销，不宣称此路径已达到原论文效率，也不为降低通信删除正确性处理。

稳态 M2C为9→12批、192→475 ct（2.47倍）；C2M为8→12批、92→305 ct（3.32倍）。增加的边界对应真实V、pre-norm残差导出及真实pre-GELU输入，见 `conversion_boundaries.csv`；未观察到8/16倍的密文数量阶跃。attention头对输出2→8 ct（4倍），特征密文1→2；197→256的padding仍参与线性/attention布局，Softmax已压缩到12×197²个有效元素，不能将其全部按256²计费。padding不足以解释M2C的120倍；它主要是**转换实际工作量差异**，不是已证明必然的原生packing成本。

**本次修复仅限统计**：两份 `layer_norm_nonlinear` 的 `sqrt comm` 调试输出原先调用推进游标的 `get_comm()`，供体本轮因此漏汇总932320 bytes。改为只读 `current_comm()` 前后差，供体头文件补同名只读接口。探针自身使用独立计数，不受此错误影响。新脚本、步长测试、报告守恒检查及两端独立构建通过；本次没有重跑12层或覆盖旧冻结清单，没有改变 Softmax/GELU、FineGrainFusion、MHP/BSGS、packing 或研究矩阵。

## M2C 同语义复测（2026-09-27，更新120倍结论）

**范围限制**：BERT为原供体五块入口，仍含占位及省略运算；ViT为真实第二层。完整 BERT 单层数值验收尚未完成，不能用本次结果替代。为遵守“只改M2C”，不顺带修复 QK/AV/Softmax/GELU。

两侧隔离探针共用同一份从现有 ViT 提取的转换代码，保持有符号语义、40位扩展余量、16个FFT保护位、40位FFT根精度及逐RNS取模；保留各自既有输入位宽/scale和HE上下文。BERT修正全零MSB假设、64位扩展上限和单素数表示的不一致，没有向ViT新增步骤。供体槽索引构造还存在共享递推变量 `pos` 的OpenMP竞争：仅将该循环改成ViT已有的顺序构造后，公开正负数样例由失败变为通过；此单点修复已落到供体 `he.h`。转换统一保留在可复跑的隔离探针中，正式 B 源码与二进制不替换。

| 指标 | BERT128供体入口 | ViT第二层 | 倍率 |
|---|---:|---:|---:|
| M2C调用批次 | 9 | 12 | 1.33 |
| 转换明文/密文形状数 | 192 | 475 | 2.47 |
| 槽元素 | 1572864 | 3997696 | 2.54 |
| 槽有符号扩展，MB | 139.423 | 396.851 | 2.85 |
| 系数有符号扩展，MB | 202.337 | 542.773 | 2.68 |
| **M2C合计，MB** | **341.760** | **939.624** | **2.75** |

两类扩展分别调用9/12批，每类元素数均为1572864/3997696；BERT位宽40→80及56→96，ViT为37→77及53→93。FFT实际调用192/475次，处理3145728/7995392个系数位置；逐RNS取模/NTT调用192/475次，处理16121856/43646976个RNS系数，二者通信均为0。B2A是扩展子项：每类576/768个worker调用，不再加到父项总额。

ViT总额中wrap检测占93.67%，B2A占6.33%。系数B2A有4.83倍的局部倍率，但其calls/元素仍为正常规模：40位COT固定负载为 `5N + N/8 + 16×worker_calls` 字节；实测额外18462048 bytes出现在第二层第19批，是按源码固定消息长度扣除后归于Ferret `rcot()` 补充的余量。它是在线开销，不移到offline，也不是额外执行一次模型转换。见 `b2a_payload_reconciliation.csv`；这是精确字节核算归因，不是新增rcot探针计数。

密文生成和发送不在 `mpc_to_ckks` 返回明文分享的函数内：BERT生成99个（96个源于M2C、3个为原占位V），发送102个；ViT生成/发送497个。ViT的475→497来自两处逆标准差明文各从1份广播为12份；BERT保留原half packing。生成本身0通信，客户端发送136.335/433.011 MB单列，不混入M2C表，也未改布局。

按[论文算法2及附录C.3/C.4](https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf)：本地Encode、系数ring-to-field、客户端加密发送、服务端加分享属于原协议；逐RNS取模是其多素数实例化。FFT前槽扩展、宽整数FFT及保护位是当前数值实现的适配，不是ViT特有，也不是算法2额外列出的独立步骤。槽分享与FFT系数分享属于不同域，两次扩展没有处理同一中间值；删除其中一次需另行证明编码与舍入等价，本轮不删除。`protocol_mapping.json`记录逐步关系；不宣称其是所有BLB实现的最低成本。

公开测试在双方各覆盖37位/scale12和40位/scale20、16384个正负槽值；客户端最大误差分别为BERT `7.75925e-5 / 3.04973e-7`、ViT `7.76853e-5 / 3.05191e-7`，全部通过。调用顺序、密文数、槽数、输入位宽/scale与前次逐项一致，父子通信守恒通过。证据目录 `outputs/m2c_audit_20260927/`。

结论：**转换级120倍异常来自不同流程的比较；同语义后为2.75倍，与2.54倍元素规模相近。** BERT分母7.833→341.760 MB，ViT940.184→939.624 MB（仅桶归属差异，整层SCI发送量未变），不是ViT获得120倍加速。真实完整BERT验收与正式基线冻结资格不因此改变。

正式接入核查：`scripts/audit_b_m2c_integration.py` 已核实当前正式 ViT M2C 与已执行932.615 s完整12层归档中的函数相同，HE/非线性依赖相同；生成的转换核心与成功的 BERT/ViT 探针逐字节相同。无需重复替换 ViT 算法。报告为 `outputs/m2c_integration_20260927/source_identity.json`，仅证明源码一致性，不是新运行 PASS。

本轮冻结前置条件尚缺真实 BERT CKKS 单层入口：`BLB/scripts/run_bert_base.sh` 默认调用 `ckks_bert_large_main`，其输入为全1，AV仍构造 fake V；另一个 `bolt_bert_large.cpp` 使用 `Bert/LIB_BERT_BOLT`，不能直接替代同一 BLB CKKS 五块路径。确认入口后先验收 BERT/ViT fixed-clear↔secure，再以相同 online send-only 口径比较 rotation/conversion/通信，最后重跑12层并记录新二进制、依赖、checkpoint、配置与结果哈希。上述验收未完成前不更新最终冻结资格，不把合成入口的2.75倍/3.31倍当作真实 BERT 对照。

入口范围更新：按用户要求只使用 `Vit-Head/BLB`，不采用旧备份的源码、脚本或二进制。当前目录有真实权重/输入和参考分享，未找到已接通并通过验收的真实 BERT CKKS 单层入口。后续基于当前 BLB 源码补齐真实输入、算子衔接及数值验收，保留原算法和安全语义；之后再进行 M2C 同语义接入与统一统计，不以 BOLT 路径或合成计时入口代替。


## 本轮补充约束与论文验收格式

**补充实验约束与验收要求**

请在执行前将以下约束合并到前述实验方案中。本轮仍然只研究头级异构机制，不开发成本感知分配器或新的转换打包方法。

**1. 补充低比例替换实验**

单 Block 在原有配置基础上增加：

- 11S/1R/0L
- 11S/0R/1L
- 10S/1R/1L

完整 ViT 在原有每层固定配置之外，增加全网仅替换 1 个 ReLU、1 个 Linear，以及两种头各替换 1 个的低比例验证。

先从低替换比例逐步增加。即使某配置准确率下降严重，也必须保留其明文实验结果，不通过调参隐藏负结果。

**2. 固定头身份与 Linear 校准**

所有头计划必须在实验前生成并保存，记录每层被替换头的真实索引。

相同数量的头替换使用预先固定的头位置规则，不调用成本感知算法。

Linear α 采用现有数学定义，仅在2000张校准集上使用一致的方法进行拟合，并在正式实验前冻结。归一化 ReLU 的零行行为、Linear 的符号与尺度规则必须与安全执行器一致。

单 Block 初测优先复用已有验证通过的编码块，并固定其编号。其他层的位置敏感性留待后续分析，不能把单个 Block 的结果视为所有层都相同。

**3. 数值正确性优先**

每种异构计划至少区分：

- 原始ViT FP32输出；
- 异构ViT的协议等价明文输出；
- 同一异构计划的真实CKKS–MPC安全输出。

实际安全执行必须与对应异构计划的协议参考比较，报告relative-L2、max-abs、预测一致性和异常值。

保持BLB–ViT公共后端、原有MHP/BSGS与FineGrainFusion；仅启用已有异构机制所必需的执行路径与选择性转换，不启用新的异构转换打包。

确认ReLU、Linear分别使用真实安全协议和真实CKKS计算，不得使用明文fallback、surrogate或客户端恢复真实激活。

数值不通过时先停止该配置的性能结论，不允许通过放宽阈值解决。

**4. 明确单Block与完整网络计时边界**

由于BLB存在算子级及跨层融合，单Block测试必须记录实际输入状态、融合边界和必要转换，避免将省略的上下游准备开销当作性能收益。

完整网络安全实验分别记录：

- 安全计算时间；
- 完整请求墙钟；
- 初始化及每请求准备开销；
- 双方发送量之和；
- CKKS→MPC和MPC→CKKS次数、批次、密文数与耗时。

所有方法固定硬件、网络、checkpoint、预处理、数值参数和输入图片。建议每种正式配置至少重复3次，交错执行基线与异构方法。

不使用单Block时间外推完整12层时间，也不将安全计算时间与完整请求墙钟混用。

**5. 避免选择偏差**

单Block的所有预设配置都要报告明文准确率。

完整12层安全推理至少覆盖四类：
全Softmax、ReLU-only、Linear-only、ReLU+Linear混合。

如计算资源不足，可从每类选择一个已固定的代表配置，但要明确选择规则、未测配置和结论适用范围，不以挑选最快结果代替系统性比较。

准确率报告相对同一子集BLB–ViT的新增损失，同时报告相对原始FP32 ViT的总损失。使用0.5、1、2个百分点作为预先声明的损失分析区间，不要求所有配置都满足预算。

当准确率结果接近预算边界时，报告逐图片配对正确性变化，不仅依赖整体Top-1。

**6. 本轮最终需要回答的科学问题**

- 减少Softmax头是否真正降低了CKKS–MPC安全推理时间？
- ReLU与Linear的精度损失和真实成本是否不同？
- 混合头的转换/布局开销是否抵消部分计算收益？
- 不同固定替换比例是否产生了准确率—真实安全时延的权衡？
- 这些结果是否足以说明后续需要研究“头身份和组合成本感知”的配置算法？

本轮只提交：
单Block消融表、完整ViT消融表、准确率损失—真实安全时延图和简短实测分析。

不执行288个单头敏感性实验，不启动成本感知分配器，不实现新转换打包。

所有结果必须来自实际执行，缺失数据保留NOT_RUN，不用估算、理论计算量或单层外推填补。

## 六、论文实验结果表格（必须按以下格式输出）

本轮必须生成以下两张核心数值表和一张结果图。表格中的“待测”必须由真实运行结果填充，不能使用估算值、单层外推值或模拟执行时间。

### 表1：单 Block 异构注意力消融实验

**Table 1. Ablation study of heterogeneous attention within a ViT block.**

| Attention configuration | Softmax heads | ReLU heads | Linear heads | Top-1 (%) | ΔAcc. (pp) | Secure block latency (s) | Communication (GB) | C2M/M2C calls |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| All Softmax | 12 | 0 | 0 | 待测 | — | 待测 | 待测 | 待测 |
| Single ReLU | 11 | 1 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Single Linear | 11 | 0 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Sparse hybrid | 10 | 1 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| ReLU replacement | 6 | 6 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Linear replacement | 6 | 0 | 6 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Mixed attention | 6 | 3 | 3 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Balanced attention | 4 | 4 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 |

**实验与填表要求：**

- 固定一个预先选定的 ViT Block，记录其层号。
- Top-1 必须由“仅替换该 Block、其余11层不变”的完整明文 ViT 计算得到，而不是单 Block 的局部指标。
- ΔAcc. 为同一选择集上相对全 Softmax BLB–ViT 的 Top-1 准确率损失，单位为百分点。
- Secure block latency 必须来自真实 CKKS–MPC 执行，包含所测融合区域必需的转换与重排。明确孤立 Block 的计时边界。
- C2M/M2C 必须分别记录逻辑转换边界、批量调用及转换密文数；表中展示批量调用次数，详细数据保存在原始结果中。
- 每个安全配置至少重复3次，并在附属结果中报告均值、标准差及数值误差。
- 固定头身份与 Linear α，不根据结果反复调整。
- 不得根据单 Block 时延推算完整12层时延。

### 表2：完整 ViT-B/16 的异构注意力消融实验

**Table 2. Accuracy and end-to-end secure inference cost of heterogeneous attention on ViT-B/16.**

| S/R/L heads | Top-1 (%) | Top-5 (%) | ΔAcc. (pp) | Secure computation (s) | Request latency (s) | Communication (GB) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 144/0/0 | 待测 | 待测 | — | 待测 | 待测 | 待测 | 1.00× |
| 143/1/0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 143/0/1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 142/1/1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 132/12/0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 132/0/12 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 120/12/12 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 96/24/24 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

**实验与填表要求：**

- 144/0/0 为冻结的 BLB–ViT 基线。
- 143/1/0、143/0/1、142/1/1 是全网仅替换少量头，用于观察最低替换比例下的变化。
- 132/12/0 和 132/0/12 表示每个 Block 分别替换1个 ReLU 或 Linear 头。
- 120/12/12 表示每层10S/1R/1L。
- 96/24/24 表示每层8S/2R/2L。
- 准确率必须在同一选择集上测量，不能将历史50,000张的准确率与本轮5,000张结果直接混合比较。
- Secure computation 为真实安全计算累计时间；Request latency 为明确计时范围的完整请求墙钟时间，两者分列。
- Speedup = 相同计时口径下BLB–ViT耗时 / 当前方法耗时。主表优先采用 Secure computation 的加速比，完整请求加速比可单列。
- 安全推理必须执行完整12层ViT，不能单层外推。
- 真实安全推理成本较高时，允许先填写所有配置的明文准确率，再选择全Softmax、ReLU-only、Linear-only、混合异构等代表配置实测完整网络；未测位置保留“—”，不得填预测值。
- 在最终正式比较中，对实测配置至少重复3次，使用相同输入、设备、网络、密码参数和计时边界。

**已有BLB–ViT历史参考值：**

- ImageNet完整50,000张 Top-1：79.878%。
- 安全计算均值：929.69 s（同一图片5次）。
- 在线通信：35.943 GB。
- 环境：同机双A100，模拟1Gbps/0.3ms LAN。

上述数字只作参考。本轮主表需要使用匹配的选择集和安全执行配置重新验证基线，不能无条件将历史数字直接填入新的配对比较。

### 图1：准确率损失与真实安全推理时间的关系

**Figure 1. Accuracy–latency trade-off of heterogeneous attention configurations.**

绘图要求：

- 横轴：Top-1 accuracy degradation relative to BLB–ViT (pp)。
- 纵轴：Measured secure inference latency (s)。
- Softmax+ReLU、Softmax+Linear和三路径混合使用不同标记。
- 每个散点表示一个实际运行的完整ViT配置。
- 标注全Softmax BLB–ViT基线。
- 可使用0.5、1.0、2.0 pp作为准确率损失参考线。
- 只绘制真实端到端安全推理结果；没有完整网络测量的配置不得用单Block估算补点。
- 单Block准确率—时延关系可另绘补充图，但不能与完整网络散点混合。

### 最终交付要求

生成以下结果文件：

1. `single_block_heterogeneous_ablation.csv`：对应表1，含完整原始测量及重复运行统计。
2. `full_vit_heterogeneous_ablation.csv`：对应表2，含头计划与样本范围。
3. `accuracy_latency_tradeoff.pdf`：对应图1。
4. `HETEROGENEOUS_ATTENTION_RESULTS.md`：用正式论文表格展示上述结果，并用简短段落说明发现。

所有结果必须注明模型、数据子集、固定头身份、α校准方式、完整运行配置以及原始日志索引。

完成本轮后停止，不开展288个单头敏感性实验，不启动成本感知分配器，不实现新的CKKS–MPC转换打包。
