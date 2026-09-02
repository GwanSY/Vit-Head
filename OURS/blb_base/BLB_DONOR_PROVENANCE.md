# 复现来源

## CPU

- 作者提交：`25e8d86a562f1cdd3a8bf3404a89cf3d0705d062`
- 唯一协议对齐：移除提交中标为 `simulate three comparisons` 的额外一次 MSB 调用，使 GELU 恢复为作者表 9 日志和 GPU 快照中的两次比较。
- 构建迁移：移除 `SCI/src/LinearHE/CMakeLists.txt` 中一次无条件、且返回值未检查的 in-source SEAL 重建；正式构建显式使用官方镜像内同版本 SEAL 4.1.1。该变更不修改协议源码，重建后的 CPU 二进制哈希与迁移前完全一致。
- VOLE-OT：启用。
- 正式构建二进制 SHA-256：`fcb618d8d501c566566defe8cea165ca9d21ad39a7135cb3831b70aa1c7f6baa`。

## GPU

- 作者提交：`d9af284b62a2dad398d2bf4928adae3ddd6cf3b2`
- 使用仓库自带 `SCI/extern/0001-more-flexible.patch` 对 Phantom 做当前 CUDA/CMake 环境兼容处理。
- VOLE-OT：启用。
- 正式构建二进制 SHA-256：`e3e695cd335d59a6d0bb2d1bfe414ad4b9a95cea62764f268eb8eae917f4fd7f`。该值与迁移前二进制不同是因为 ELF RUNPATH 从 `.paper_repro/gpu/build2` 迁移为正式的 `build/gpu`；协议源码和运行指标保持一致。

CPU 与 GPU 提交从 `1c6ad41e146150cbd5b0af10ab75e74a97d794cb` 后分叉，不能在不改写实现的情况下合并成单一源码快照。正式目录因此保留 `cpu/` 与 `gpu/` 两套来源，并移除了迁移前失效的 Git worktree 指针。

原始迁移目录在重构时保存为：

`/home/dm/mind/Vit-Head/BLB.pre_formal_20260826`

该备份不参与构建或运行。
