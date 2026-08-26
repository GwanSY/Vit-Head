# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.linen.linear import Array

# transformers 4.31 still evaluates this annotation at import time, while
# JAX >= 0.4 removed the old alias.  Keep the pinned demo dependencies working
# without changing any model operation or value.
if not hasattr(jnp, "DeviceArray"):
    jnp.DeviceArray = jax.Array

import transformers.modeling_flax_utils as tf_flax_utils
from transformers import (
    BertConfig,
    BertTokenizerFast,
    FlaxBertForSequenceClassification,
    FlaxRobertaForSequenceClassification,
    RobertaTokenizerFast,
)

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd
import spu.utils.simulation as ppsim

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True


def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument(
        "-c", "--config", default="examples/python/llm/flax_bert/2pc.json"
    )
    parser.add_argument("--model", default="bert", type=str)
    parser.add_argument("--checkpoint", default="bert-base-uncased", type=str)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument(
        "--backend",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Bumblebee HE backend (default: gpu; GPU failure never falls back)",
    )
    parser.add_argument("--dump-pphlo", action="store_true")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="compare clear and secure logits plus all BERT hidden states",
    )
    parser.add_argument(
        "--fixed-clear",
        action="store_true",
        help="also run the official REF2K fixed-point clear diagnostic",
    )
    return parser.parse_args()


args = parse_args()
os.environ["SPU_BUMBLEBEE_HE_BACKEND"] = args.backend

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


@contextmanager
def hijack(msg: str, enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    # The official BumbleBee README asks users to replace the Flax BERT
    # activation pointer with a jax.nn.gelu call.  Recent Transformers keeps
    # that pointer in modeling_flax_utils.ACT2FN, so replacing only
    # jax.nn.gelu/fnn.gelu leaves the captured partial(nn.gelu) in place.
    tf_gelu = tf_flax_utils.ACT2FN["gelu"]
    jnn.gelu = intrinsic.f_gelu
    fnn.gelu = intrinsic.f_gelu
    tf_flax_utils.ACT2FN["gelu"] = intrinsic.f_gelu
    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    tf_flax_utils.ACT2FN["gelu"] = tf_gelu


def run_on_cpu(model, input_ids, attention_masks, labels, return_hidden_states=False):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        outputs = model(
            input_ids,
            attention_masks,
            params=params,
            output_hidden_states=return_hidden_states,
            return_dict=False,
        )
        if return_hidden_states:
            return outputs[0], outputs[1]
        return outputs[0]

    start = time.time()
    outputs = eval(params, input_ids, attention_masks)
    end = time.time()
    if return_hidden_states:
        logits, hidden_states = outputs
        print(f"CPU runtime: {(end - start)}s\noutput logits: {logits}")
        return np.asarray(logits), tuple(np.asarray(x) for x in hidden_states)
    print(f"CPU runtime: {(end - start)}s\noutput logits: {outputs}")
    return np.asarray(outputs)


def run_on_spu(
    model,
    input_ids,
    attention_masks,
    labels,
    dump_pphlo=False,
    return_hidden_states=False,
):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        with hijack("gelu", enabled=True):
            outputs = model(
                input_ids,
                attention_masks,
                params=params,
                output_hidden_states=return_hidden_states,
                return_dict=False,
            )
        if return_hidden_states:
            return outputs[0], outputs[1]
        return outputs[0]

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_attention_masks = ppd.device("P1")(lambda x: x)(attention_masks)
    spu_params = ppd.device("P1")(lambda x: x)(params)
    spu_eval = ppd.device("SPU")(eval, copts=copts)
    if dump_pphlo:
        pphlo = spu_eval.dump_pphlo(spu_params, spu_input_ids, spu_attention_masks)
        lines = pphlo.splitlines()
        for idx, line in enumerate(lines):
            if '"pphlo.exponential"' in line:
                print(f"PPHLO exponential context at line {idx + 1}:")
                for context_line in lines[max(0, idx - 3) : min(len(lines), idx + 18)]:
                    print(context_line)
        return
    start = time.time()
    outputs_spu = spu_eval(spu_params, spu_input_ids, spu_attention_masks)
    end = time.time()
    outputs_spu = ppd.get(outputs_spu)
    if return_hidden_states:
        logits_spu, hidden_states_spu = outputs_spu
        print(f"SPU runtime: {(end - start)}s\noutput logits: {logits_spu}")
        return np.asarray(logits_spu), tuple(np.asarray(x) for x in hidden_states_spu)
    print(f"SPU runtime: {(end - start)}s\noutput logits: {outputs_spu}")
    return np.asarray(outputs_spu)


def run_on_fixed_clear(model, input_ids, attention_masks, return_hidden_states=True):
    """Run the same graph with SPU's official fixed-point debug protocol.

    REF2K is the upstream ring^2k reference backend: it keeps fixed-point
    arithmetic but has no security guarantee and directly exposes the first
    reconstructed share.  This is a diagnostic only; the benchmark remains
    on the BUMBLEBEE backend.
    """
    print("Running on fixed-clear (official REF2K diagnostic) ...")
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.REF2K,
        field=spu_pb2.FieldType.FM64,
        fxp_fraction_bits=18,
        fxp_exp_mode=0,
        fxp_exp_iters=6,
        experimental_enable_colocated_optimization=True,
        experimental_disable_mmul_split=True,
    )
    sim = ppsim.Simulator(2, config)

    def eval(params, input_ids, attention_masks):
        with hijack("gelu", enabled=True):
            outputs = model(
                input_ids,
                attention_masks,
                params=params,
                output_hidden_states=return_hidden_states,
                return_dict=False,
            )
        if return_hidden_states:
            return outputs[0], outputs[1]
        return outputs[0]

    spu_eval = ppsim.sim_jax(sim, eval, copts=copts)
    start = time.time()
    outputs = spu_eval(model.params, input_ids, attention_masks)
    end = time.time()
    if return_hidden_states:
        logits, hidden_states = outputs
        outputs = np.asarray(logits), tuple(np.asarray(x) for x in hidden_states)
    else:
        outputs = np.asarray(outputs)
    print(f"fixed-clear runtime: {(end - start)}s\noutput logits: {outputs[0] if return_hidden_states else outputs}")
    return outputs


def compare_correctness(clear_outputs, secure_outputs):
    clear_logits, clear_hidden_states = clear_outputs
    secure_logits, secure_hidden_states = secure_outputs

    def error_stats(clear, secure):
        delta = np.asarray(secure, dtype=np.float64) - np.asarray(clear, dtype=np.float64)
        return np.max(np.abs(delta)), np.mean(np.abs(delta)), np.sqrt(np.mean(delta * delta))

    max_error, mean_error, rms_error = error_stats(clear_logits, secure_logits)
    print(
        "correctness logits: "
        f"max_abs={max_error:.8g} mean_abs={mean_error:.8g} rms={rms_error:.8g}"
    )
    if len(clear_hidden_states) != len(secure_hidden_states):
        raise RuntimeError(
            f"hidden-state count mismatch: clear={len(clear_hidden_states)} "
            f"secure={len(secure_hidden_states)}"
        )
    for layer, (clear, secure) in enumerate(zip(clear_hidden_states, secure_hidden_states)):
        max_error, mean_error, rms_error = error_stats(clear, secure)
        print(
            f"correctness hidden[{layer:02d}]: "
            f"max_abs={max_error:.8g} mean_abs={mean_error:.8g} rms={rms_error:.8g}"
        )


def main(tokenizer_func, model_func, checkpoint, max_length):
    dataset = load_dataset("glue", "cola", split="test")
    model = model_func.from_pretrained(checkpoint)
    tokenizer = tokenizer_func.from_pretrained(checkpoint)

    print(
        f"Model: {checkpoint}, architecture={model.config.model_type}, "
        f"hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}, "
        f"labels={model.config.num_labels}, max_length={max_length}"
    )

    for dummy_input in dataset:
        features, labels = dummy_input["sentence"], dummy_input["label"]

        encoded = tokenizer(
            features,
            return_tensors="jax",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]
        print(
            f"Input: sentence={features!r}, input_ids.shape={input_ids.shape}, "
            f"attention_mask.sum={int(attention_masks.sum())}"
        )

        if args.correctness:
            clear_outputs = run_on_cpu(
                model,
                input_ids,
                attention_masks,
                labels,
                return_hidden_states=True,
            )
            secure_outputs = run_on_spu(
                model,
                input_ids,
                attention_masks,
                labels,
                return_hidden_states=True,
            )
            compare_correctness(clear_outputs, secure_outputs)
            if args.fixed_clear:
                fixed_clear_outputs = run_on_fixed_clear(
                    model, input_ids, attention_masks, return_hidden_states=True
                )
                print("fixed-clear versus secure:")
                compare_correctness(fixed_clear_outputs, secure_outputs)
        else:
            run_on_cpu(model, input_ids, attention_masks, labels)
            run_on_spu(model, input_ids, attention_masks, labels, args.dump_pphlo)
        break  # just test one sentense


if __name__ == "__main__":
    tokenizer = BertTokenizerFast
    model = FlaxBertForSequenceClassification
    main(tokenizer, model, args.checkpoint, args.max_length)
