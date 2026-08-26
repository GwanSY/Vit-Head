# BOLT

简洁中文运行命令、BERT/ViT 模型选择、CPU/GPU 后端区别和文件功能表见
[scripts/run/readme.md](scripts/run/readme.md)。

中文 ViT 端到端密态推理、加密方法和构建运行说明见 [README_VIT_ZH.md](README_VIT_ZH.md)。

Implementation of **BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers**
([IEEE S&P 2024](https://doi.org/10.1109/SP54263.2024.00130)).

This repo bundles the SCI 2PC library (from the [EzPC `bert` branch](https://github.com/Clive2312/EzPC/tree/bert));
SEAL and Eigen are git submodules and build automatically.

---

## 1. Install dependencies (Ubuntu/Debian)

```bash
sudo apt-get install g++ cmake make libgmp-dev libssl-dev
```

Requires **g++ ≥ 8** (C++17). OpenMP ships with g++. `git` with network access is needed to fetch SEAL/Eigen.

## 2. Get the code

```bash
git clone --recursive https://github.com/cmu-cryptosystems/BOLT.git
cd BOLT
# if you already cloned without --recursive:
git submodule update --init --recursive
```

## 3. Build

```bash
cd SCI
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=./install ..
cmake --build . --target install --parallel
```

Binaries land in **`SCI/build/bin/`**.

## 4. Get the model weights

Download the prepared `weights_txt/` bundle for your task from Google Drive:

**https://drive.google.com/drive/u/1/folders/13bBok39UevQ-6hDWHtBVtLJrYVo5VnsR**

Place it under a directory of your choice so the files live at `<data>/weights_txt/…`.
`BOLT_BERT` loads the model from `<path>/weights_txt/` and reads each sample's
`inputs_<i>_data.txt` / `inputs_<i>_mask.txt` from the same folder.

`<data>` is the task directory whose **direct child is `weights_txt/`**.  It is
not the ZIP path and not the `weights_txt/` directory itself.  For example,
after extracting the official `bolt.zip` in this checkout, MRPC uses:

```text
/home/dm/mind/MIND_code/Pencil/BOLT/datasets/official_bert/bolt/quantize/mrpc
└── weights_txt/
    ├── classifier.weight.txt
    ├── classifier.bias.txt
    ├── inputs_0_data.txt
    └── inputs_0_mask.txt
```

The downloaded `bolt.zip` used for the local reproduction is 817,672,642
bytes with SHA256
`f1eba7db14148798df407192a2043fecde212c9e20128c879a3c969a258bdbed`.

Before opening any network connection, validate all 196 model tensors and the
requested inputs with:

```bash
python3 scripts/preflight_bert_data.py \
  --data datasets/official_bert/bolt/quantize/mrpc \
  --role both --id 0 --num-sample 1
```

## 5. Run secure inference

Run one process per party (two terminals on one host, or two machines).

**Server — ALICE (holds the weights):**
```bash
cd SCI/build/bin
./BOLT_BERT r=1 path=<data> num_sample=1
```

**Client — BOB (holds the input, connects to the server):**
```bash
cd SCI/build/bin
./BOLT_BERT r=2 ip=<server-ip> path=<data> output=<results.txt> num_sample=1
```

On a single machine use `ip=127.0.0.1`. The default port is `8000` (the run also uses `8001`).

For this checkout, the reproducible single-host wrapper validates data and
launches both parties.  BERT opens the selected port plus 64 nonlinear
channels, so the wrapper defaults to the non-ephemeral range `18000–18064`:

```bash
TARGET=BOLT_BERT scripts/build.sh
BERT_DATA="$PWD/datasets/official_bert/bolt/quantize/mrpc" \
  scripts/run_bert_e2e.sh
```

Logs and the final result are written under `outputs/bert_e2e/` by default.

## BOLT GPU entry points

The GPU targets keep BOLT's BFV protocol and use only BLB's low-level Phantom
evaluator. They do not use BLB's model implementation, and IRON remains on its
original CPU/SEAL path.

```bash
TARGET=BOLT_BERT_GPU scripts/build.sh
TARGET=BOLT_VIT_GPU scripts/build.sh

CUDA_VISIBLE_DEVICES=0 BERT_DATA=<data> scripts/run_bert_gpu.sh
CUDA_VISIBLE_DEVICES=0 scripts/run_vit_gpu.sh <image> <output> <port>
```

The GPU runner requires the Phantom package produced by
`../BLB/repro/build_vit_b16.sh` with `USE_HE_GPU=ON`; it is restricted to device
0 and checks the `backend=phantom_bfv protocol=BOLT` log marker.

### With word-elimination optimization (input pruning)

Word elimination is built into `BOLT_BERT` and enabled with `prune=1` (point `path` at the pruned-weights bundle):

```bash
./BOLT_BERT r=1 prune=1 path=<prune-data> num_sample=1                                        # server
./BOLT_BERT r=2 prune=1 ip=<server-ip> path=<prune-data> output=<results.txt> num_sample=1    # client
```

### Common flags

| Flag | Meaning |
|------|---------|
| `r` | Role: `1` = server (ALICE), `2` = client (BOB) — **required** |
| `path` | Directory containing `weights_txt/` — **always set** |
| `ip` | Server address (client only) |
| `p` | Port (also uses `p+1`) |
| `num_sample` | Number of samples to run |
| `id` | Index of the first sample |
| `prune` | Word elimination / input pruning (`1`/`0`) |
| `output` | Output file (client only) |

---

## Plaintext accuracy reference

`bert.py` runs the quantized model in the clear against the same weights to report accuracy.
Pass the data root with `--data_dir` — it reads `<data_dir>/{quantize,prune}/<task>/weights_txt/`:

```bash
pip install numpy torch transformers tqdm scipy scikit-learn
```

```bash
python bert.py --data_dir <data> --task_name mrpc --sample_num 100   # add --online_prune for word elimination
```

## Citation

```bibtex
@INPROCEEDINGS{bolt24,
  author={Pang, Qi and Zhu, Jinhao and Möllering, Helen and Zheng, Wenting and Schneider, Thomas},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)}, 
  title={BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers}, 
  year={2024},
  pages={4753-4771},
  doi={10.1109/SP54263.2024.00130}}
```
