# Table 6：独立 HE-only 精度测试

这个测试只测 CKKS HE，不执行 HE→MPC、MPC→HE、Softmax、GeLU 或 LayerNorm 的
MPC 部分，因此不会把 conversion/FFT 误差算进论文 Table 6 的 HE MSE。

## 参数审计

```bash
cd /workspace/BLB
cmake -S SCI -B build_blb_formal/sci
cmake --build build_blb_formal/sci --target ckks_table6_he_only -j2
build_blb_formal/sci/bin/ckks_table6_he_only mode=audit
```

输出同时列出：

- `current_constructor_chain`：`Linear_ckks::set_HE` 传给 SEAL 的完整链；
- `seal_key_level_chain`：SEAL key level；
- `seal_first_data_chain`：首个可用于 ciphertext 的 data level；
- `paper_reported_chain`：论文 Table 6 原样参数。

当前审计结论是 Block 1、2 的声明链与论文不一致，Block 3、4、5 一致。参数审计
成功只表示读到了当前真实配置，不表示五块 MSE 已通过。

## Block 1 的 FFN2 HE-only 子图

先开服务端（ALICE，模型方）：

```bash
cd /workspace/BLB
build_blb_formal/sci/bin/ckks_table6_he_only \
  mode=block1 r=1 p=65400 \
  model=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
  oracle=/workspace/BLB/outputs/bert_oracle_public_mask_v3
```

再开客户端（BOB）：

```bash
cd /workspace/BLB
build_blb_formal/sci/bin/ckks_table6_he_only \
  mode=block1 r=2 p=65400 ip=127.0.0.1 \
  model=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
  oracle=/workspace/BLB/outputs/bert_oracle_public_mask_v3
```

该模式使用真实 layer-0 `fgf4_ln_ffn_gelu` fixed-clear 向量和真实
`output.dense` 权重/bias，调用生产代码的 `params_preprocessing_ct_pt` 和
`linear_2` BSGS 路径，解密最终 FFN2 输出后与 dense clear counterpart 比较。
oracle 的两份 share 只在这个离线公开精度测试中重构；正式 secure inference 不调用
此测试，也不恢复中间明文。

当前范围必须按输出理解：

- `TABLE6_HE_ONLY ... stage=ffn2_matmul`：真实 FFN2 ct-pt MatMul 子图的 MSE；
- `TABLE6_BLOCK1 status=BLOCKED`：尚未把 residual add 和 LayerNorm HE 子图纳入
  独立测试，所以不能把 FFN2 的数值冒充论文完整 Block 1 MSE；
- `TABLE6_BLOCKS_2_TO_5 status=BLOCKED`：其余完整 HE-only 子图尚未接入；
- `conversion_count=0`：所有输出误差均不含 HE↔MPC conversion。

论文 Table 6 的门槛是各完整 block 相对 plaintext counterpart 的 MSE `< 1e-11`；
论文报告的 Block 1 MSE 是 `7.2e-13`，不是本测试应硬编码得到的结果。
