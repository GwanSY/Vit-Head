# BERT-base 第 0 层 BLB 验收报告

> 2026-08-05 更新：官方 BLB 容器中的 MRPC BERT-base 12 层权重和 128×768 输入已复制到 `checkpoints/official-blb-mrpc/weights_txt`，shape/packing/SHA256 manifest PASS；layer 0 已按官方 `bert.py` 的 post-norm 顺序接入真实输入、两组 LayerNorm 参数、attention/FFN residual 和三 oracle 比较入口。下文早期 checkpoint/输入 BLOCKED 是历史记录；当前阻塞是官方容器五块 MSE 实测和 pooling/classifier。

日期：2026-08-04  
唯一工作根目录：`/home/dm/mind/MIND_code/Pencil/BLB`  
总状态：`BLOCKED`  
`formal_ready=no`

本报告只统计 `SCI/tests/bert_large_bolt` 的 BLB 正式 BERT 路径；旧独立 ViT
smoke target 已删除，任何历史 smoke 结果均不计入本报告。

## 实际执行命令

```bash
cd /home/dm/mind/MIND_code/Pencil/BLB

CMAKE_BIN=/usr/bin/cmake \
CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
CUDA_ARCH=70 JOBS=2 \
bash repro/build_formal_blb.sh

bash repro/check_source.sh
bash repro/audit_formal_blb.sh

/usr/bin/python3 repro/prepare_bert_checkpoint.py \
  --raw-dir checkpoints/bert-base/raw \
  --output-dir checkpoints/bert-base/weights_txt \
  --layers 0

PORT=46000 OUTPUT="$PWD/outputs/bert_layer0_formal" \
  bash repro/run_original_bert.sh

build_blb_formal/sci/bin/ckks_bert_large_main \
  r=1 p=46000 manifest_verified=0 layers=1

build_blb_formal/sci/bin/ckks_bert_large_main \
  r=1 p=46000 manifest_verified=1 layers=1 \
  model=checkpoints/bert-base/weights_txt

/usr/bin/python3 repro/accept_single_layer.py \
  outputs/bert_layer0_formal
```

checkpoint 下载命令也已尝试，但本机配置的代理 `127.0.0.1:7897` 拒绝连接，取消代理后的 Hugging Face 直连超时。没有将不完整下载或其他工程文件作为 checkpoint。

## 验收矩阵

| 验收项 | 状态 | 证据或阻塞原因 |
|---|---|---|
| 仅在 BLB 内修改、构建和产出日志 | PASS | 正式构建根为 `build_blb_formal/sci`；正式 runner 对模型路径做 BLB 根目录边界检查 |
| 正式路径拒绝 fake weight、零 mask、模拟残差 | PASS（代码门禁） | `CORRECTNESS_GATE placeholders=0`；fake V、编码零份额、自相加 residual 和临时 LayerNorm `w/b` 已从活动正式路径移除；真实输入/oracle 编译期门禁不可由参数绕过 |
| Algorithm 1 CKKS→MPC | PASS（代码门禁） | 当前 ciphertext level 的 RNS limbs 使用 SEAL CSPRNG 和 `sample_poly_uniform`；mask 不经过 `CKKSEncoder`；随后调用真实 `field_to_ring` |
| `pp_1..pp_4` 与真实 BSGS 权重接入 | PASS（代码门禁） | 四个预处理入口启用；缺权重或 shape 不匹配直接抛错；device BSGS 未审计分支 fail-closed |
| 正式目标编译 | PASS | `ckks_bert_large_main` 构建成功，构建脚本仍明确输出 `formal_ready=no` |
| checkpoint shape/packing/SHA256 manifest | PASS | 官方 MRPC 12 层 text checkpoint 已在 BLB 内生成并验证 shape/packing/SHA256 manifest |
| 真实 BERT-base 第 0 层权重 | PASS | `checkpoints/official-blb-mrpc/weights_txt`；缺项仍 fail-closed |
| 真实 layer-0 输入及 residual | PASS（代码接线） | `inputs_0_data.txt`/`inputs_0_mask.txt` 由 BOB 私有输入；post-norm attention 的原始输入 residual、以及 FFN2 所需的真实 ln1 输出 residual 都以安全份额传递；尚未完成两方实测 |
| FP32、fixed-clear、secure 三路 oracle | PARTIAL | `prepare_bert_oracle.py`/`secure_stage_mse` 已实现并通过离线 manifest 校验；日志中尚无官方容器五块可验收 B↔C 数值 |
| Figure 10 五块 B↔C MSE `<1e-11` | BLOCKED | 五块尚无 secure 两方实测；严格验收器拒绝缺失记录 |
| 旋转、ct-ct/ct-pt、转换、截断、通信、延迟 | BLOCKED | 计数入口已接入，但完整两方执行未发生，不能报告有效数值 |
| 单层安全性和正确性 | BLOCKED | 上述 checkpoint、输入和 oracle 条件未满足 |
| 扩展到 metadata 的 12 层 | BLOCKED（有意禁止） | 当前执行深度固定为 1；单层 PASS 前 `layers != 1` 会被正式门禁拒绝 |
| pooling/classifier | BLOCKED（未伪造） | 正式 CKKS 路径未接通，保持 `formal_ready=no` |

## 实际日志

- `outputs/bert_layer0_build_20260804.log`：构建 PASS、`formal_ready=no`。
- `outputs/bert_layer0_source_check_20260804.log`：源码完整性和正式占位门禁 PASS。
- `outputs/bert_layer0_audit_20260804.log`：Algorithm 1、`pp_1..pp_4`、符号和 smoke 隔离 PASS；checkpoint BLOCKED；总状态 BLOCKED。
- `outputs/bert_checkpoint_prepare_20260804.log`：因缺 `raw/config.json` 按设计 BLOCKED。
- `outputs/bert_checkpoint_download_20260804.log`：官方 Hugging Face 地址短连接测试超时。
- `outputs/bert_layer0_run_gate_20260804.log`：因缺 BLB 内 checkpoint，在启动两方前 BLOCKED。
- `outputs/bert_layer0_direct_gate_20260804.log`：未验证 manifest 的直接启动在联网前 BLOCKED。
- `outputs/bert_layer0_input_gate_20260804.log`：历史日志；当时因真实 layer-0 输入未实现而在联网前 BLOCKED。
- `outputs/bert_layer0_acceptance_20260804.log`：严格验收器对缺失两方正式日志返回 BLOCKED。
- `outputs/bert_layer0_git_scope_20260804.log`：本轮 git 状态、diff 统计、嵌套 Phantom 状态和正式二进制 SHA256。

## Git diff 范围

变更只位于 BLB git 根目录，范围分为：

- 正式 BERT/CKKS：`SCI/tests/bert_large_bolt/`；
- CKKS/Phantom 目录内兼容与构建：`SCI/src/LinearHE/`、`SCI/extern/phantom/`、`SCI/cmake/`、`SCI/src/CMakeLists.txt`；
- MPC 截断统计：`SCI/src/BuildingBlocks/truncation.{h,cpp}`；
- 构建、checkpoint、审计和严格验收：`repro/`；
- 中文说明与验收证据：`README.md`、`REPRODUCTION_STATUS.md`、`docs/`；
- 旧 ViT smoke/BOLT/BFV adapter 已删除，不属于当前正式验收路径。

逐文件状态和统计以 `outputs/bert_layer0_git_scope_20260804.log` 为准。

## 下一解除条件

当前不需要再下载另一个 checkpoint：官方 MRPC text checkpoint 已通过 manifest。下一步必须在容器内生成并验证 `oracle_shares`、重编译当前 BLB，然后实测五路安全 oracle；在五个融合块全部达到 fixed-clear↔secure MSE `<1e-11` 且安全、统计门禁通过之前，结果保持 `BLOCKED`，不得扩到 12 层。
