# 复现状态

日期：2026-08-05

总状态：`formal_ready=no`，单层 BERT-base 正式验收 `BLOCKED`。

已完成：

- 正式源码占位模式静态门禁 PASS；
- `pp_1..pp_4` 启用，QKV/WO/FFN 的 BSGS 接入预处理权重；
- CKKS→MPC 改为当前 RNS 环的 CSPRNG 均匀 mask，并真实调用 `field_to_ring`；
- 原零 mask、fake V 和自相加模拟残差不再作为正式执行路径；
- checkpoint 转换、shape/packing/SHA256 manifest 和缺项拒绝脚本完成；
- 正式目标目录内构建 PASS；
- 未验证 manifest、缺权重、层数大于 1 均会 fail-closed。
- 真实 post-norm layer-0 输入、两组 LayerNorm 参数和 residual 已接入；三 oracle 的 BLB-local 参考份额、manifest 校验和 MPC 标量 MSE 代码已接入，在联网/OT 前 fail-closed；LayerNorm 缺真实 `w/b` 份额也会拒绝执行。
- 末端 ln2 已补上仓库既有 host CKKS 仿射收尾（`layer_norm_affine_ckks`），运行日志必须出现 `ckks_affine=1` 才能通过严格验收。

阻塞：

- 官方 Zenodo 容器中的 12 层 MRPC BERT-base checkpoint、pooler/classifier 和 128×768 输入已经复制到 BLB，manifest PASS；第 0 层 post-norm 真实输入、两组 LayerNorm 参数和 attention/FFN residual 已接入 secure main；
- FP32、fixed-clear、secure 三 oracle 的离线参考和安全比较入口已实现，但尚未在官方容器中跑出五块 `ORACLE_STAGE`；
- Figure 10 五块尚无 `<1e-11` 的两方实测 MSE；
- pooling/classifier 未接通，故 `formal_ready=no`；
- 单层未 PASS，禁止扩展到 12 层。

tiny ViT 结果只算 `formal_result=no` 链路测试。

## 官方 artifact / 容器补充（2026-08-05）

- Zenodo V1 源码、V2 容器、V3 Phantom 补丁的文件大小和 MD5 已核对；目录内下载/断点续传/校验脚本完成；
- 官方容器的导入脚本和只挂载当前 BLB 的运行脚本完成，容器内 `/home/CKKS-MPC` 不作为正式源码或依赖；
- 宿主机沙箱外 GPU 检查 PASS：8×NVIDIA A100 80GB，驱动 560.35.03；
- Zenodo 单文件下载仍受失效代理阻塞，但阿里云作者镜像已成功拉取，digest `sha256:f696ecc441daeb1e05daaee6b955e12cdc5ea0cbb1f5335dab57f769167d20b4`；
- Docker/GPU/mount preflight PASS：容器内可见 8×A100，当前 BLB 只读挂载为 `/workspace/BLB`；
- 官方 MRPC checkpoint 导入及 12 层 manifest PASS。权限长期生效仍需重新登录或在当前 shell 执行 `newgrp docker`。

## 容器单层实测（2026-08-05）

- `outputs/audit_real_layer0_20260805.log`：`BLB_FORMAL_AUDIT status=PASS`。
- `outputs/container_formal_real_layer0_20260805.log`：旧二进制两方均在联网前输出 `FORMAL_GATE status=BLOCKED reason=three_oracle_not_implemented`；该日志不代表当前 oracle 代码。
- `outputs/accept_real_layer0_20260805.log`：`SINGLE_LAYER_ACCEPTANCE status=BLOCKED reason=missing three-oracle result for fgf1_ln_attention`。
- `outputs/build_formal_oracle_gpu_20260805.log`：当前三 oracle 代码的 GPU 目标构建 PASS，二进制仍标记 `formal_ready=no`。
- `outputs/audit_after_oracle_infra_20260805.log`：当前 BLB 内静态审计 PASS；运行时仍要求 `oracle_shares/manifest.json` 和五块安全 MSE。
- `outputs/oracle_preprocess_official_20260805.log`、`outputs/oracle_verify_official_20260805.log`：官方 layer-0 五块参考份额生成和校验 PASS。
- `outputs/audit_oracle_official_20260805.log`：当前官方份额存在时的 BLB 静态审计 PASS。
- `outputs/build_formal_oracle_gpu_postnorm_privacy_20260805.log`：post-norm、BOB 私有输入和真实 ln1 residual 路径构建 PASS。
- `outputs/audit_postnorm_header_fix_20260805.log`：当前版本静态审计 PASS。
- `outputs/host_formal_header_fix_20260805.log`：主机双进程因沙箱 `bind: Operation not permitted` BLOCKED；未计入正式结果。
- `outputs/accept_host_header_fix_20260805.log`：严格验收因缺少五块实际 `ORACLE_STAGE` BLOCKED。
- `outputs/build_formal_oracle_gpu_final2_20260805.log`、`outputs/audit_formal_blb_final2_20260805.log`：ln2 CKKS 仿射收尾后的构建和静态审计 PASS；安全两方仍未实测。
- `outputs/audit_final_after_docs_20260805.log`、`outputs/verify_oracle_final_continuation_20260805.log`、`outputs/build_incremental_final_continuation_20260805.log`：本轮文档后静态审计、oracle 校验和增量构建均 PASS。
- `outputs/host_formal_latest_20260805.log`：主机短时双进程仍因系统 `bind: Operation not permitted` BLOCKED；`outputs/accept_host_formal_latest_20260805.log` 因缺少 `ORACLE_STAGE` BLOCKED。
- `outputs/direct_gate_layers2_latest_20260805.log`：未完成单层验收时请求 2 层被 fail-closed 拒绝。
- 以上不是正式复现结果；没有生成五块 MSE，因此不能扩展到 12 层。
