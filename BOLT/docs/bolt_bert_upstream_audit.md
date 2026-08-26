# BOLT 官方 BERT 复现审计

## 复现范围

本审计以 CMU Cryptosystems 的官方 BOLT `main` 分支为基线，先运行
12 层 BERT，再继续 ViT。完整 BERT 使用官方 `bolt.zip` 中的
`quantize/mrpc`，没有用 tiny fixture 代替。

官方数据包：

- 文件：`datasets/official_bert/bolt.zip`
- 字节数：`817672642`
- SHA256：`f1eba7db14148798df407192a2043fecde212c9e20128c879a3c969a258bdbed`
- MRPC `<data>`：`datasets/official_bert/bolt/quantize/mrpc`
- 结构：`<data>/weights_txt/...`

## 已确认的上游问题

1. `SCI/tests/bert_bolt/bert_utils.cpp` 声明了 `classifier.bias.txt`，但
   上游实际使用 `bp_fname` 读取 `bert.pooler.dense.bias.txt`。这会将
   pooler bias 的末尾两个值误作二分类 bias。现已改为读取
   `classifier.bias.txt`。
2. 上游默认 `path` 和 `output` 是作者机器上的绝对路径。现改为空数据
   路径和仓库内可用的默认输出名；调用者必须显式给出 `path`。
3. 上游缺文件时读取函数返回空向量，后续才可能越界崩溃。现有入口在
   建立网络连接前检查 ALICE 的 196 个模型文件、BOB 的输入/mask，模型
   加载后再次检查每个张量的精确形状。
4. 命令行接受任意 `num_class`，但实现由 `NUM_CLASS=2` 静态编译。现对
   非 2 的值 fail-closed。
5. 同机运行需要主端口加 64 个 nonlinear 通道。在 Linux 临时客户端
   端口区间选择主端口可能发生碰撞；`scripts/run_bert_e2e.sh` 默认使用
   `18000–18064`。

## 共享后端生产兼容问题

首次完整运行在第 0 层 QK 处发现：原 `Linear::linear_1` 将“QK 输出
前缀 + 保留的 V 密文尾部”作为一个合并 buffer 传入 cross-packing
内核，共享兼容层却要求 buffer 大小必须恰好等于 QK 输出。修复位于
`SCI/src/BOLTTransformer/bolt_linear.cpp`：要求 buffer 至少容纳 QK
前缀，并只按实际生成的 QK 前缀统计 ciphertext 数。计算循环、packing、
rotation、multiply、relinearization 均未改变。

该生产形状已加入 `BERT_SHARED_BACKEND_REGRESSION`。

## 构建和运行

```bash
TARGET=BOLT_BERT scripts/build_vit.sh

python3 scripts/preflight_bert_data.py \
  --data datasets/official_bert/bolt/quantize/mrpc \
  --role both --id 0 --num-sample 1

scripts/run_bert_shared_backend_regression.sh

BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc" \
  scripts/run_bert_e2e.sh
```

也可以按官方方式开两个终端；`path` 指向 MRPC 任务目录，不是
`weights_txt` 本身。

## 当前验证结果

`outputs/regression/bert_shared_backend.csv`：

| op | original hash | shared hash | ciphertexts | rotations | status |
|---|---|---|---:|---:|---|
| bert_ct_pt_bsgs | `6dc7338deabdece0` | `6dc7338deabdece0` | 1 / 1 | 4 / 4 | PASS |
| bert_qk_cross | `308d2269cca704ec` | `308d2269cca704ec` | 1 / 1 | 896 / 896 | PASS |
| bert_softmax_v | `4c8472adb97fea81` | `4c8472adb97fea81` | 1 / 1 | 128 / 128 | PASS |

完整 MRPC BERT 的最终状态必须以 `outputs/bert_e2e*/results.csv` 和双方
日志为准；失败或未完成时不得写 `BOLT_BERT_E2E=PASS`。
