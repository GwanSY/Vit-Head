# Flax BERT Example
## Config

We use the checked-in config file `flax_bert/2pc.json`.

```json
"runtime_config": {
    "protocol": "BUMBLEBEE",
    "field": "FM64",
    "enable_pphlo_profile": true,
    "enable_hal_profile": true,
    "fxp_exp_mode": 0,
    "fxp_exp_iters": 6,
    "experimental_enable_colocated_optimization": true,
    "experimental_disable_mmul_split": true
}
```

This example demonstrates how to use SPU to run private inference on a pre-trained BERT model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Launch the GPU backend (default)

    ```sh
    examples/python/llm/flax_bert/run_openbumblebee_2pc.sh
    ```

   This selects the visible GPU with the most free memory, builds with
   `--config=gpu`, and runs with `--backend gpu`. PhantomFHE must be installed
   at `/opt/openbumblebee-phantom`. A missing CUDA device or Phantom build is a
   hard error; the runtime never falls back to SEAL.

3. Run the preserved CPU/SEAL backend explicitly

    ```sh
    examples/python/llm/flax_bert/run_openbumblebee_2pc.sh --backend cpu
    ```

4. Reproduce the colocated 1 Gbps / 0.3 ms RTT condition

    ```sh
    examples/python/llm/flax_bert/run_openbumblebee_2pc.sh --shape-network
    ```

The direct driver also defaults to GPU and accepts `--backend cpu`:

```sh
bazel run -c opt --config=gpu //examples/python/llm/flax_bert -- \
  --backend gpu --max-length 128 \
  --config `pwd`/examples/python/llm/flax_bert/2pc.json
```
