# 运行说明

> **2026-10-08 当前轮次：仅头级异构机制消融。** 执行前必须遵守 [方案首节](EXPERIMENT_PLAN.md) 与其文末表格要求。本轮只用 B/H 和固定头计划，后文 costs/offline、HA/HAP、新打包命令属于历史说明，禁止作为本轮队列运行。2026-10-09 用户已更正数据授权：直接将现有验证集按类别划分为 2000/5000/43000 张，最后 43000 张本轮不评估；无需训练集。固定的 16 个计划及状态见 [本轮契约](blb_base/outputs/heterogeneous_attention_20261009/experiment_contract.json)。本轮新增用户指定的 `HETEROGENEOUS_ATTENTION_RESULTS.md` 不受下文历史“两份 Markdown”约定限制。

实验定义及后续实现任务见 [实验实现计划](EXPERIMENT_PLAN.md)。以下命令均在 `OURS/blb_base` 执行；版本迁移已通过构建与接口回归，B 已完成单图完整密态推理；数据集级准确率及 H/HA/HAP 全网实验仍待验收。

## 1. 环境与资产检查

```bash
cd /home/dm/mind/Vit-Head/OURS/blb_base
python3 -B scripts/run_vit.py --show-config
python3 -B ours_vit_b16/scripts/verify_vit_manifest.py \
  --model checkpoints/vit_b16/bolt_model.txt \
  --manifest checkpoints/vit_b16/manifest.json
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv
```

需要 Python 3.10+、NumPy、Pillow、Docker/NVIDIA 容器支持；构建需要 CMake ≥3.17、CUDA 12.3。默认镜像为 `ours:restored-fast-20260903d`，主程序及 Phantom 库由宿主构建产物覆盖。启动前确认镜像、本地二进制及动态库存在：

```bash
docker image inspect ours:restored-fast-20260903d --format '{{.Id}}'
ls build/container-gpu/bin/ours_vit_b16_main \
   build/container-gpu/extern/phantom/lib/libPhantom.so
```

模型、输入目录是指向 `Vit-Head/BLB` 的符号链接，不要删除链接目标。固定模型为 `google/vit-base-patch16-224`；数据为 `Vit-Head/datasets/imagenet2012/val`，标签清单是无表头的 `validation_manifest.csv`，每行 `images/文件名.JPEG,类别编号`。

需要更新构建时执行（复用现有构建树及工具链）：

```bash
JOBS=8 bash scripts/build_vit_b16.sh
```

## 2. 运行现有 B 基线

入口默认读 `ours_vit_b16/experiments/experiment.json`，默认 `secure.version=B`、全部 Softmax。唯一版本为 B/H/HA/HP/HAP/J/BP；旧版本参数与名称均拒绝接受。`run_vit.py` 默认执行 `secure`，完整 12 层加分类头。

下面的 `0 1` 是物理 GPU 示例，请换成两张空闲卡，并在同组对照中固定。默认要求每卡空闲显存 ≥70000 MiB、利用率 ≤5%；网络为容器内 1 Gbps / 0.3 ms RTT。

```bash
python3 -B scripts/run_vit.py --version B --gpus 0 1 \
  --output-dir outputs/matrix/B_r01 --print-command
python3 -B scripts/run_vit.py --version B --gpus 0 1 \
  --output-dir outputs/matrix/B_r01
```

每次换新输出目录，如 `B_r02`。终端仅显示启动/结束，详细进度与结果：

```bash
tail -f outputs/matrix/B_r01/run.log
# 完成后查看：
cat outputs/matrix/B_r01/last_run.json
cat outputs/matrix/B_r01/native/validation.json
```

`native/` 保存双方日志、`run_metadata_party*.txt`、`execution_totals_party*.csv`、`secure_logits_party*.txt`、`reference.json` 和网络证据。`numerical_pass` 是单输入数值验证，不是数据集准确率；退出成功也不等于所有性能门槛通过。`acceptance_pass` 仅检查当前版本的执行和数值；性能须与同环境实测 B 或配对来源比较，已删除历史固定秒数门槛。实验根目录还保存 `config.secure.json`、`plan_binding.json`，禁止覆盖同一安全运行根目录。

## 3. 自定义输入与配置

配置支持 `//` 注释；配置内相对路径均相对于 `blb_base/`。复制后修改，避免覆盖默认配置：

```bash
mkdir -p outputs/matrix
cp ours_vit_b16/experiments/experiment.json outputs/matrix/config.json
python3 -B ours_vit_b16/scripts/preprocess_vit_image.py \
  ../../datasets/imagenet2012/val/images/ILSVRC2012_val_00001001.JPEG \
  --manifest checkpoints/vit_b16/manifest.json \
  --output outputs/matrix/image_1001.txt
```

编辑 `outputs/matrix/config.json`：

| 参数 | 设置方法 |
|---|---|
| `paths.output_dir` | 本次实验根目录，例 `outputs/matrix/experiment_01` |
| `secure.input` | 例 `outputs/matrix/image_1001.txt` |
| `secure.gpu_pair` | 例 `[0,1]`；默认 `[0,1]`；`null` 才共享 `gpu_id`，不用于正式双卡比较 |
| `secure.version` | B/H/HA/HP/HAP/J/BP，也可用 `--version` 覆盖 |
| `secure.assignment` | 新格式离线 assignment JSON；校验版本和 checkpoint，不覆盖请求的版本 |
| `secure.head_plan` | H/HA/HP/HAP/J 使用的固定计划；B/BP 留空，使用内置全 S |
| `secure.paired_config` | HP/HAP 必填，对应 H/HA 已运行的 `config.secure.json`；同目录须有完整 `plan_binding.json` |

```bash
python3 -B scripts/run_vit.py --config outputs/matrix/config.json \
  --gpus 0 1 --output-dir outputs/matrix/custom_r01 --print-command
# 检查命令后，去掉 --print-command 执行。
```

## 4. 现有离线选头与成本入口

校准/测试默认各 1000 张，预算 `[0.1,0.5,1.0]` pp，主预算 0.5 pp。HA 使用无 P 成本，J 使用有 P 成本；为 J 设置独立成本 CSV 和输出目录，禁止混用。先查看命令：

```bash
python3 -B scripts/run_vit.py --config outputs/matrix/config.json \
  --version HA --stage costs --print-command
python3 -B scripts/run_vit.py --config outputs/matrix/config.json \
  --version HA --stage offline --print-command
```

先准备校准 α 对应的探针清单和计划（`make_composition_probe_suite.py --help`），设置 `paths.probe_manifest`；成本资产准备好后依次执行：

```bash
python3 -B scripts/run_vit.py --config outputs/matrix/config.json --version HA --stage costs
python3 -B scripts/run_vit.py --config outputs/matrix/config.json --version HA --stage offline
```

`costs` 会启动真实安全测量，耗时较长；设备来源检查已统一为配置中的固定 GPU 对，历史旧字段成本表不会自动迁移。`offline` 不启动安全推理，但会做完整明文评估。缺少合格实测成本时会阻断，不会自动补造结果。

离线产物位于 `<paths.output_dir>/offline/`：`budget_candidates.csv`、`budgets/<预算>/head_assignment.json`、`head_plan.txt`、校准/测试清单、敏感度及成本验证表。将选定 assignment 路径填入配置后，可用第 3 节入口做单次安全验证。使用同一入口逐版本运行；没有自动替你批跑完整网络。

## 5. 从 HA 运行 HAP

完成 HA 运行后，复制该次 `config.secure.json`，设置 `secure.version="HAP"`、`secure.paired_config="outputs/matrix/HA_r01/config.secure.json"`，保留 assignment/头计划、输入与执行参数；运行时指定新的输出根目录：

```bash
python3 -B scripts/run_vit.py --config outputs/matrix/HAP.json \
  --version HAP --output-dir outputs/matrix/HAP_r01 --print-command
# 核对后去掉 --print-command。
```

HP 同理，来源版本为 H。程序比较原始头身份、协议和 α 的语义哈希、冻结资产哈希及 GPU/网络/线程等设置，并复用来源 Docker image ID；只有物理重排不改变语义时才可配对。HP/HAP 禁止执行 offline 搜索。旧分组打包实现已并入 P，共用同一异构计算路径；数值等价性和收益仍须由实际实验确认。

## 6. 文档与历史记录

`OURS` 递归仅保留本文件和 `EXPERIMENT_PLAN.md`。整理前 197 份 Markdown（含旧说明、依赖说明及历史结果摘要）已完整备份到 `../document_archives/ours_markdown_20260920/markdown.tar.gz`，逐文件 SHA-256 在同目录 `manifest.json`；此次版本迁移保留历史原始结果与许可证文件。

只查看旧文档可在 `blb_base/` 执行：

```bash
tar -tzf ../../document_archives/ours_markdown_20260920/markdown.tar.gz
# 例：读取旧头计划格式说明，不恢复到工作目录。
tar -xOzf ../../document_archives/ours_markdown_20260920/markdown.tar.gz \
  blb_base/HEAD_PLAN_FORMAT.md
```

当前异构汇总只输出 CSV；人工维护文档维持两份。

旧版本源码迁移前备份：`../document_archives/ours_version_migration_20260920.tar.gz`。已删除旧专属驱动、旧候选矩阵和 Linear 的 MPC surrogate 分支；保留历史原始实验结果，未把它们改名为新版本证据。

实验定义固定在 `blb_base/ours_vit_b16/scripts/experiment_contract.py`（数学函数、安全目标、离线选头和成本/最优性边界）。新 assignment 自动包含定义及哈希，缺失或不匹配会拒绝执行；旧 assignment 请通过原冻结配置重新导出并核对语义计划哈希，不能只补标签冒充兼容。正式 HA→HAP 必须沿用同一份合格 assignment，不重搜或重拟合。新增成本字段在旧日志缺失时为空（NA），需重新采样才能获得；采集不修改历史结果。

性能排查：新报告的 `new_heterogeneous_packing=false` 只表示未启用矩阵中的 P。旋转与编码的逐算子归因见 `native/block_online_profile_party1.csv`；双方时间与通信须一起看。比较相同层数的两份 native 目录（单次配对只作诊断）：

```bash
python ours_vit_b16/scripts/audit_b_performance.py \
  --before outputs/matrix/<before>/native \
  --after outputs/matrix/<after>/native \
  --output outputs/matrix/<after>/performance_audit
```

更换本次修复的二进制后，成本表和 HA/HAP 正式配对需绑定同一新构建；不能用旧 HA 与新 HAP 报告纯打包收益。

本次已通过的 B 完整回归位于 `outputs/matrix/B_rotation_reuse_20260920_r01/`，数值结果见 `native/validation.json`，旋转/编码热点见 `performance_audit/operator_totals.csv`。

只读评估固定权重缓存容量（不分配缓存、不启动 GPU、不修改 B）：

```bash
python ours_vit_b16/scripts/audit_weight_cache.py \
  --run outputs/matrix/B_rotation_reuse_20260920_r01/native \
  --output outputs/performance_audit_20260921/weight_cache
```

输出容量估算 CSV/JSON；系统缓存方案及计时边界见实验计划第 6 节，当前尚未实现热缓存推理。

## Strict BLB-ViT：已冻结的正确性产物

2026-09-22 的源码/协议及数值验收产物已冻结，清单为 `blb_base/ours_vit_b16/experiments/b_baseline_freeze.json`。完整12层、GPU 2/5、端云双进程、1 Gbps / 0.3 ms RTT；clear/fixed/secure Top-1 均248，双方 logits 完全相同，secure/fixed 相对L2=1.73366%、maxabs=0.137451，在线1505.218348 s。仅为单图回归，不是数据集准确率；**性能合理性仍为 NEEDS_AUDIT，该耗时不作为最终性能冻结结论**。旧802.777 s保留为历史实现结果。

在本文件前述 `blb_base` 工作目录下，复跑已冻结产物：

```bash
python3 -B scripts/run_vit.py \
  --config outputs/strict_blb_20260922/strict_baseline_config.json \
  --stage secure --version B --gpus 2 5 \
  --output-dir outputs/matrix/B_strict_repeat_r01
```

该配置固定二进制与 Phantom 副本，不依赖后续覆盖的默认 build 文件。使用新输出目录；GPU 可用性由脚本检查。重新编译后的二进制必须重新验收，不能继承旧冻结哈希。

证据在 `blb_base/outputs/strict_blb_20260922/strict_full_r02/`：`native/validation.json` 为数值验收，`native/strict_protocol_review.json` 为源码/协议审阅，`strict_review/strict_validation.json` 为全部门槛 PASS。`executed_artifacts/` 保存实际二进制、库及源码快照。转换逐步关系见上一级 `algorithm2_restoration.json`。运行时 `candidate_not_frozen` / verification=0 是构建时标签；事后冻结状态以绑定同一二进制的严格审计报告为准，不修改原始运行日志。

复核当前源码与构建的冻结绑定：

```bash
python3 -B scripts/validate_b_migration.py \
  --vit-run outputs/strict_blb_20260922/strict_full_r02/native \
  --output outputs/strict_blb_20260922/strict_full_r02/strict_review
```

源码或构建变化后，此命令拒绝沿用旧 PASS；历史冻结产物仍保留供复跑。

## B 性能退化审计（2026-09-27）

证据目录：`blb_base/outputs/shape_audit_20260927/`。`audit_summary.json` 汇总结论，`operator_counts.csv` 比较同机 BERT128/ViT256 的实际 HE API 调用，`conversion_boundaries.csv` 记录每次转换密文数，`paired_block_timing.csv` 记录单层时间。155.2392 s 是历史 BERT 单层云端计时×12，通信量含补算，不能当作当前同版本完整 BERT 实测。供体有占位 V/GELU 输入和省略的 QK/AV 运算，详见实验计划。

本次仅修复指数截断的串行退化及重复截断；等价关系见 `repair_equivalence.json`，相对冻结源码的完整改动见 `implementation_repair.patch`。模型、端云协议、GELU、Softmax 多项式、转换和新方法禁用条件保持不变。完整网络结果以 `strict_repair_full/native/validation.json` 为准，不能继承旧二进制的冻结资格。

修复后完整12层已通过：在线 **932.614689 s**（15.54 min），clear/fixed/secure Top-1均248，双方logits相同，相对L2=1.73056%、maxabs=0.155518；在线双方发送量33.56085 GB。相对历史1505.218348 s少38.04%，但GPU编号不同且各一次，不作为稳定配对加速比。性能最终冻结仍待审，旧冻结清单不覆盖。

复跑修复产物及重新汇总（在 `blb_base` 下，输出使用新目录）：

```bash
python3 -B scripts/run_vit.py --config outputs/shape_audit_20260927/reproduce_config.json \
  --stage secure --gpus 0 1 --output-dir outputs/shape_audit_20260927/recheck
python3 -B scripts/summarize_b_shape_audit.py
```

计数副本保存在 `bert_count/` 和 `vit_count_before_r02/`，包括源码清单、独立二进制及日志；不覆盖正式构建。新采样使用 `scripts/prepare_b_count_audit.py --output <新目录>`，ViT 加 `--vit`；生成的 `build.sh` 需在既有构建容器中执行。计数是源码 HE API 调用（含零步旋转调用），不是 CUDA kernel 次数；旧手工计数遗漏了部分 rescale/非线性乘法，不与新表混用。

## Rotation / 通信专项复核（2026-09-27）

结果在 `blb_base/outputs/wire_rotation_audit_20260927/`：`rotation_by_operator.csv`、`communication_by_category.csv` 是主表，`rotation_sites.csv` 提供源码位置和步长，`conversion_boundaries.csv` 提供转换次数/密文数；原始双方法日志在 `bert/run/`、`vit/run/`。这里是 BERT 测试单层与 ViT 前两层探针，不是再次跑完整12层。

在 `blb_base` 下重建报告（只读日志）：

```bash
python3 -B scripts/report_b_wire_audit.py
```

新探针用 `python3 -B scripts/prepare_b_wire_audit.py --output <新目录>` 生成；ViT 加 `--vit`。生成的 `build.sh` 在既有构建容器内运行，独立产物不覆盖正式构建。ViT 两层采样命令如下（输出目录必须另取新名）：

```bash
OURS_VIT_PHYSICAL_GPUS=0,1 \
OURS_VIT_BINARY_OVERRIDE="$PWD/outputs/wire_rotation_audit_20260927/vit/build/bin/ours_vit_b16_main" \
bash scripts/run_ours_container.sh B ours_vit_b16/experiments/all_softmax_head_plan.txt \
  outputs/wire_rotation_audit_repeat/vit/run 2 block
```

专项结论：旋转 API 仍为13.35倍，但供体省略真实 QK/AV 运算；统一发送口径后首层通信5.72倍、稳态层5.52倍，主要来自 MPC→CKKS 的120.04倍。详见实验计划。当前只修复 `sqrt comm` 日志推进统计游标的问题；未改算法、安全参数或原冻结产物。修复后的隔离编译在 `postfix_compile/`，不能据此沿用旧二进制的数值验收。

## M2C 同语义复测（接续上述专项）

结果：同一有符号转换流程下，BERT供体五块入口/ViT第二层的 M2C 为 **341.760 / 939.624 MB，2.75倍**；槽元素数2.54倍。原120倍主要来自转换流程不一致，不能当作 ViT 加速成果。**BERT仍不是完整真实模型验证**，本轮没有改其占位 QK/AV/GELU 等路径。

证据目录 `blb_base/outputs/m2c_audit_20260927/`：`phase_summary.csv` 为阶段调用数/元素/字节，`share_conversion_summary.csv` 为子项，`ciphertext_generation.csv` 与 `ciphertext_transport.csv` 分离生成和发送，`audit_summary.json` 汇总结论。`bert_r04/run/`、`vit_r03/run/` 保存成功日志；`bert_r03/` 保存槽索引竞争导致的失败样例。槽索引顺序构造修复已落到供体 `he.h`。BERT 使用统一 M2C 的改动仍在隔离探针中；ViT 正式 B 原本就使用该转换核心，本轮未替换其二进制。

在 `blb_base` 下汇总现有结果或完整复跑（新目录、64个SCI通道、GPU0/1）：

```bash
python3 -B scripts/report_m2c_audit.py
bash scripts/run_m2c_audit.sh outputs/m2c_repeat_r01 0 1
```

正式 B 的 M2C 源码一致性复核（不运行 GPU、不构成新的数值验收）：

```bash
python3 -B scripts/audit_b_m2c_integration.py --output outputs/m2c_integration_20260927/source_identity.json
```

当前仅使用 `Vit-Head/BLB`，不使用旧备份。`scripts/run_bert_base.sh` 的 CKKS 入口仍有全1输入和 fake V；`bolt_bert_large.cpp` 虽读取真实输入，但链接 `LIB_BERT_BOLT`，不能代替同一路径的 BLB CKKS 对照。真实 layer-0 权重与五块参考分享存在，但未找到已接通并通过数值验收的真实 BERT CKKS 单层入口。后续须在当前源码中补齐该入口与验收，再统一统计、复测12层和冻结；不能将材料校验当作运行 PASS。

真实 BERT 单层候选（验收通过前不得用于最终表格）：从 `Vit-Head` 运行下面命令。入口在 `BLB/bert_base/real_bert_layer.cpp`，与 ViT 共用当前 Strict B 内核、同语义 M2C 和 send-only 计数器；不使用旧备份。构建及日志放在独立输出目录，保留正式 B 二进制。

```bash
bash BLB/scripts/run_real_bert_layer.sh outputs/real_bert_new/r01 0 1
```

程序保存 MRPC 验证样本的两方阶段分享；`BLB/scripts/validate_real_bert_layer.py` 在受信任的离线验证环境重构并比较，不得用于生产用户输入。`validation.json` 检查九个边界的 fixed/secure 相对 L2≤0.03（或绝对误差≤2/4096），同时记录 clear/fixed 偏差。在线通信不包含分享文件写入及离线验证，不把 COMPLETE 当作数值 PASS。

复跑脚本使用现有容器及构建依赖，先执行公开正负数转换测试；数值失败立即停止，不生成通过报告。初始化、测试样例和已执行的离线OT不计入在线；在线按需OT补充仍计入。
