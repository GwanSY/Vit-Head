"""Reproducible paired, real two-party tests of graph-only head optimizations."""
import argparse
import copy
import hashlib
import json
import itertools
import random
import subprocess
import time
from pathlib import Path

from runtime import ROOT, command, digest, write
from experiment import case_config, run_case

CASES = {"S10R2": "S"*10+"R"*2, "S6R6": "S"*6+"R"*6,
         "S6R3L3": "S"*6+"R"*3+"L"*3}
VARIANTS = ("baseline", "organized", "batched", "reciprocal_only")


def paired_worker(out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from spu import spu_pb2
    import spu.utils.distributed as ppd
    import spu.utils.frontend as frontend
    from transformers.models.vit.modeling_flax_vit import FlaxViTLayerCollection
    from model import load, secure_gelu
    from optimized_heads import bind_variant, lower_reciprocal_only
    ppd.logger.setLevel(30)
    meta = json.loads((out / "run.json").read_text()); c = meta["config"]
    model, pixels = load(Path(meta["model"]), Path(meta["image"]))
    full = c["task"] == "vit"
    if full:
        params, input_array = model.params, pixels
        plan = c["head_plan"]
        def apply(params, x):
            return model(x, params=params, train=False).logits
    else:
        original = model(pixels, params=model.params, train=False, output_hidden_states=True)
        input_array = np.asarray(original.hidden_states[0], dtype=np.float32)
        params = {"0": model.params["vit"]["encoder"]["layer"]["0"]}
        config = copy.deepcopy(model.config); config.num_hidden_layers = 1
        layer = FlaxViTLayerCollection(config, dtype=jnp.float32)
        plan = dict(heads=[c["types"]], alpha=[c["alpha"]])
        def apply(params, x):
            return layer.apply({"params": params}, x)[0]
    variants = c["paired_variants"]
    def function(variant, secure):
        def compute(params, x):
            # Native packing comparisons deliberately keep the complete
            # original PPHLO graph, including its public gathers/reciprocals.
            graph_variant = "baseline" if c.get("native_packing_study") else variant
            with bind_variant(plan, graph_variant):
                if secure:
                    with secure_gelu():
                        return apply(params, x)
                return apply(params, x)
        return compute
    references = {v: np.asarray(function(v, False)(params, jnp.asarray(input_array)))
                  for v in dict.fromkeys(["baseline"]+variants)}
    equivalence = {v: float(np.max(np.abs(ref-references["baseline"])))
                   for v, ref in references.items()}
    # Explicit inverse * numerator differs from direct FP32 division by a few
    # ULPs, although both secure graphs use the same inverse and multiply. This
    # reference-only roundoff check does not relax any secure numerical gate.
    plaintext_acceptance = {v:bool(np.allclose(ref,references["baseline"],atol=5e-6,rtol=1e-6))
                            for v,ref in references.items()}
    if not all(plaintext_acceptance.values()):
        raise RuntimeError(f"Plaintext semantics changed: {equivalence}")
    ppd.init(meta["spu"]["nodes"], meta["spu"]["devices"])
    device = ppd.device("SPU")
    owned_params = ppd.device("P1")(lambda x: x)(params)
    secure_params = jax.tree_util.tree_map(lambda x: ppd.Device.move(x, device), owned_params)
    owned_input = ppd.device("P2")(lambda x: x)(input_array)
    saved_colocated = device.runtime_config.experimental_enable_colocated_optimization
    if not full: device.runtime_config.experimental_enable_colocated_optimization = False
    try: secure_input = ppd.Device.move(owned_input, device)
    finally: device.runtime_config.experimental_enable_colocated_optimization = saved_colocated
    def storage(server, values):
        return [spu_pb2.ValueMetaProto.FromString(x.spu_share.meta).storage_type for x in values]
    flat = jax.tree_util.tree_leaves(secure_params)
    input_storage = [client.run_return(storage, [secure_input.refs[rank]])
                     for rank, client in enumerate(device.node_clients)]
    weight_storage = [client.run_return(storage, [x.refs[rank] for x in flat])
                      for rank, client in enumerate(device.node_clients)]
    if not full and any("ashr" not in s.lower() for row in input_storage for s in row):
        raise RuntimeError("Block input must remain shared")
    if any("priv" not in s.lower() for row in weight_storage for s in row):
        raise RuntimeError("Weights must remain private")
    copts = spu_pb2.CompilerOptions(); copts.enable_optimize_denominator_with_broadcast = True
    secure = {v: device(function(v, True), copts=copts) for v in variants}
    compilation = {}; active = None
    original_compile = frontend.compile
    compiler_dir = out / "compiler"; compiler_dir.mkdir()
    def audited(*args, **kwargs):
        start = time.perf_counter()
        executable, tree = original_compile(*args, **kwargs)
        executable.code = lower_reciprocal_only(executable.code,active)
        code = executable.code
        record = dict(sha256=hashlib.sha256(code).hexdigest(), gelu_calls=code.count(b"f_gelu"),
                      reciprocal_calls=code.count(b'"pphlo.reciprocal"'),
                      gather_calls=code.count(b'"pphlo.gather"'), compile_s=time.perf_counter()-start)
        if record["gelu_calls"] != (12 if full else 1):
            raise RuntimeError("Intrinsic GELU count changed")
        previous = compilation.get(active)
        if previous and previous["sha256"] != record["sha256"]:
            raise RuntimeError("Executable changed between paired calls")
        compilation[active] = record
        (compiler_dir / f"{active}.mlir").write_bytes(code)
        return executable, tree
    frontend.compile = audited
    vectors = {f"reference_{v}": ref for v, ref in references.items()}
    meta.update(plaintext_equivalence_max_abs=equivalence,plaintext_equivalence_pass=plaintext_acceptance,
                plaintext_gate="FP32 abs <= 5e-6 + 1e-6*abs(reference); secure FM64/f18 and numerical gates unchanged",
                logical_head_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                input_sha256=hashlib.sha256(input_array.tobytes()).hexdigest(),
                input_storage_types=input_storage, weight_storage_types=weight_storage,
                measurement_unit="native maximum of fixed P0/P1 total execution; request includes reveal",
                outputs_file="outputs.npz")
    execution = 0
    schedule = [(0, v, True) for v in c["warmup_order"]]
    schedule += [(pair["pair"], v, False) for pair in c["pair_schedule"] for v in pair["order"]]
    try:
        for pair_number, variant, warmup in schedule:
            active = variant
            if c.get("native_packing_study"):
                def select_native_mode(server, selected):
                    import os
                    mode = "1" if selected == "tree_batch" else "0"
                    os.environ["SPU_BB_PACK_TREE_BATCH"] = mode
                    return mode
                acknowledgements = [client.run_return(select_native_mode,variant)
                                    for client in device.node_clients]
                if acknowledgements != (["1","1"] if variant=="tree_batch" else ["0","0"]):
                    raise RuntimeError("Both parties must select the same packing mode")
            start = time.perf_counter()
            actual = np.asarray(ppd.get(secure[variant](secure_params, secure_input)))
            request = time.perf_counter()-start
            reference = references["baseline"]
            delta = actual.astype(np.float64)-reference.astype(np.float64)
            relative = float(np.linalg.norm(delta)/np.linalg.norm(reference))
            maximum = float(np.max(np.abs(delta)))
            finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
            numeric = finite and relative <= c["gate"]["relative_l2_max"] and maximum <= c["gate"]["max_abs_max"]
            if full: numeric = numeric and int(actual.argmax()) == int(reference.argmax())
            result = dict(repeat=pair_number, pair=pair_number, variant=variant,
                execution_index=execution, native_execution=execution+1, warmup=warmup,
                request_s=request, compile_s=compilation[variant]["compile_s"],
                relative_l2=relative, max_abs=maximum, finite=finite, numerical_pass=numeric,
                reference_top1=int(reference.argmax()) if full else None,
                secure_top1=int(actual.argmax()) if full else None,
                compiled_sha256=compilation[variant]["sha256"])
            if c.get("native_packing_study"):
                result["native_packing_mode_ack"] = acknowledgements
            vectors[f"e{execution+1}_{variant}"] = actual
            np.savez_compressed(out / "outputs.npz", **vectors)
            meta.setdefault("warmups" if warmup else "measurements", []).append(result)
            meta["compilations"] = compilation
            write(out / "run.json", meta)
            print(json.dumps(result), flush=True)
            if not numeric: raise RuntimeError("Secure numerical validation failed")
            execution += 1
    finally: frontend.compile = original_compile


def correctness_and_compile(out):
    """Offline numerical/IR checks. These are never timing samples."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from spu import spu_pb2
    import spu.utils.frontend as frontend
    import heads
    from optimized_heads import organized, batched, reciprocal_only
    from model import load, secure_gelu
    from optimized_heads import bind_variant, lower_reciprocal_only
    from transformers.models.vit.modeling_flax_vit import FlaxViTLayerCollection
    rng = np.random.default_rng(20261014)
    edge_checks = []
    for types in ("S"*10+"R"*2, "S"*6+"R"*6, "S"*6+"R"*3+"L"*3, "RSLSRSLSRSLS"):
        alpha = np.linspace(.001, .012, 12, dtype=np.float32).tolist()
        for name in ("random", "zeros", "negative", "large"):
            q, k, v = [rng.normal(size=(1,12,7,64)).astype(np.float32) for _ in range(3)]
            if name == "zeros": q *= 0
            if name == "negative": q[:] = -1; k[:] = 1
            if name == "large": q *= 8
            args = [jnp.asarray(x) for x in (q,k,v)]
            reference = np.asarray(heads.attention(*args, list(types), alpha))
            for variant, fn in (("organized", organized), ("batched", batched), ("reciprocal_only",reciprocal_only)):
                actual = np.asarray(fn(*args, list(types), alpha))
                error = float(np.max(np.abs(reference-actual)))
                if error > 1e-6 or not np.isfinite(actual).all():
                    raise RuntimeError(f"Attention edge case failed: {types}/{name}/{variant}/{error}")
                edge_checks.append(dict(types=types, input=name, variant=variant, max_abs=error))
    model, _ = load(Path("/work/BOLT/checkpoints/vit_b16_hf"),
                    Path("/work/datasets/imagenet2012/val/images/ILSVRC2012_val_00039905.JPEG"))
    config = copy.deepcopy(model.config); config.num_hidden_layers = 1
    layer = FlaxViTLayerCollection(config, dtype=jnp.float32)
    params = {"0": model.params["vit"]["encoder"]["layer"]["0"]}
    args = (jax.tree_util.tree_map(lambda x: np.zeros(x.shape,x.dtype),params),
            np.zeros((1,197,768),np.float32))
    flat, _ = jax.tree_util.tree_flatten(args)
    alpha = json.loads((ROOT / "out/07/alpha.json").read_text())["alpha"][0]
    copts = spu_pb2.CompilerOptions(); copts.enable_optimize_denominator_with_broadcast = True
    destination = out / "compiler"; destination.mkdir(exist_ok=True)
    records = []
    for case, types in CASES.items():
        plan = dict(heads=[list(types)], alpha=[alpha])
        for variant in VARIANTS:
            def compute(params, x):
                with bind_variant(plan, variant), secure_gelu():
                    return layer.apply({"params":params}, x)[0]
            executable, _ = frontend.compile(frontend.Kind.JAX, compute, args, {},
                [f"in{i}" for i in range(len(flat))], [spu_pb2.Visibility.VIS_SECRET]*len(flat),
                lambda xs: [f"out{i}" for i in range(len(xs))], copts=copts)
            code = lower_reciprocal_only(executable.code,variant)
            (destination / f"{case}-{variant}.mlir").write_bytes(code)
            record = dict(case=case, variant=variant, sha256=hashlib.sha256(code).hexdigest(),
                gather_calls=code.count(b'"pphlo.gather"'),
                reciprocal_calls=code.count(b'"pphlo.reciprocal"'),
                gelu_calls=code.count(b"f_gelu"))
            if record["reciprocal_calls"] != (1 if variant in ("batched","reciprocal_only") else 2):
                raise RuntimeError(f"Normalization batching did not compile as intended: {record}")
            if record["gather_calls"] != ((4 if "L" in types else 3) if variant=="baseline" else 0):
                raise RuntimeError(f"Public selection did not compile as intended: {record}")
            records.append(record); print(record, flush=True)
    write(out / "preflight.json", dict(status="pass", edge_cases=edge_checks,
          compiled=records, note="Offline checks only; no secure performance observations"))


def launch_preflight(out):
    out.mkdir(parents=True, exist_ok=True)
    merged = command(["docker","inspect","openbumblebee","--format","{{.GraphDriver.Data.MergedDir}}"])
    inside = "/work/"+str(out.relative_to(ROOT.parents[1]))
    args = ["docker","run","--rm","--network","none","--name","ours-bb-opt-preflight",
            "--cpuset-cpus","25-26","--memory","16g","-w","/work/OURS/bumblebee",
            "-v",f"{ROOT.parents[1]}:/work:ro","-v",f"{out}:{inside}:rw",
            "-v",f"{ROOT}/build:/opt/bb:ro","-v",f"{ROOT}/build/stage2.so:/opt/bb/spu/libspu.so:ro",
            "-v",f"{ROOT}/build/phantom:/opt/openbumblebee-phantom:ro",
            "-v","/usr/local/cuda-12.3:/opt/cuda:ro",
            "-v",f"{merged}/tmp/openbumblebee-pydeps-py310:/tmp/openbumblebee-pydeps-py310:ro"]
    env = dict(PYTHONPATH="/tmp/openbumblebee-pydeps-py310:/opt/bb:/work/OURS/bumblebee",
        JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",
        SPU_BUMBLEBEE_HE_BACKEND="cpu",SPU_BB_ENABLE_MMUL_PACK="1",SPU_BB_SET_IEQUAL_BITS="16",
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",YACL_NUM_THREADS="1",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        LD_LIBRARY_PATH="/opt/openbumblebee-phantom/lib:/opt/cuda/lib64")
    for key,value in env.items(): args += ["-e",f"{key}={value}"]
    subprocess.run(args+["ours-bb:base","python","optimize_study.py","preflight-worker","--out",inside],check=True)


def run_blocks(out, phase, sessions, pairs, variants):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "preflight.json").exists(): raise RuntimeError("Run preflight first")
    alpha = json.loads((ROOT / "out/07/alpha.json").read_text())["alpha"][0]
    rng = random.Random(20261014 + (1 if phase=="confirm" else 0))
    design_path = out / f"{phase}_design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text())
        if (design["sessions"],design["pairs"],design["variants"]) != (sessions,pairs,variants):
            raise RuntimeError("Frozen study design differs")
    else:
        cases = []
        balanced_orders = {case:list(itertools.permutations(variants)) for case in CASES}
        for orders in balanced_orders.values(): rng.shuffle(orders)
        for session in range(1,sessions+1):
            order = list(CASES); rng.shuffle(order)
            for case in order:
                permutation = list(balanced_orders[case][(session-1)%len(balanced_orders[case])])
                warmup_order = list(variants); rng.shuffle(warmup_order)
                schedule = []
                for pair in range(1,pairs+1):
                    sequence = permutation if pair%2 else permutation[::-1]
                    schedule.append(dict(pair=pair, order=list(sequence)))
                cases.append(dict(session=session,case=case,warmup_order=warmup_order,schedule=schedule))
        design = dict(phase=phase,sessions=sessions,pairs=pairs,variants=variants,records=cases,
                      seed=20261014+(phase=="confirm"),statistical_unit="fresh runtime, paired means")
        write(design_path,design)
    for entry in design["records"]:
        name = f"{phase}-r{entry['session']:02}-{entry['case']}"
        folder = out / phase / "log" / name
        config = case_config("block","cpu",types=list(CASES[entry["case"]]),alpha=alpha,
            layout="joint",organization="paired",native_library=str(ROOT / "build/stage2.so"),
            trace_intervals=True,repeats=1,measured_calls=pairs*len(variants),
            paired_variants=variants,warmup_order=entry["warmup_order"],pair_schedule=entry["schedule"],
            plan_name=entry["case"],session=entry["session"],phase=phase,study="optimization")
        if folder.exists() and json.loads((folder / "run.json").read_text())["status"]=="complete":
            if json.loads((folder / "run.json").read_text())["config"] != config:
                raise RuntimeError("Cached execution configuration differs")
            continue
        run_case(out / phase,name,config)
        print(f"Completed {name}",flush=True)


def run_full(out, rounds, variant, case="S10R2"):
    """Cold AB/BA pairs: each arm gets its own independent fresh 2PC runtime."""
    analysis = json.loads((out / "analysis.json").read_text())
    eligible = [r for r in analysis["summary"] if r["phase"]=="confirm" and
                r["case"]==case and r["variant"]==variant and
                r["saved_ci_low_s"]>0 and r["p_holm"]<.05]
    if not eligible:
        raise RuntimeError("Full inference requires clear, independently confirmed Block benefit")
    alphas = json.loads((ROOT / "out/07/alpha.json").read_text())["alpha"]
    plan = dict(heads=[list(CASES[case]) for _ in range(12)],alpha=alphas)
    design_path = out / "full_design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text())
        if (design["rounds"],design["variant"],design["case"])!=(rounds,variant,case): raise RuntimeError("Frozen full design differs")
    else:
        orders = [["baseline",variant],[variant,"baseline"]]*(rounds//2)
        if rounds%2: orders.append(["baseline",variant])
        random.Random(20261015).shuffle(orders)
        records = [dict(round=i+1,order=order) for i,order in enumerate(orders)]
        design = dict(rounds=rounds,variant=variant,case=case,plan=plan,records=records,
            trigger=eligible[0],scope="12-layer ViT; cold independent runtimes within randomized paired rounds",
            source=f"Same logical {case} heads in every layer; all layer-specific original alpha values retained")
        write(design_path,design)
    for record in design["records"]:
        for arm in record["order"]:
            name = f"full-r{record['round']:02}-{case}-{arm}"
            folder = out / "full/log" / name
            config = case_config("vit","cpu",head_plan=plan,layout="joint",organization="paired",
                native_library=str(ROOT / "build/stage2.so"),trace_intervals=True,repeats=1,measured_calls=1,
                paired_variants=["baseline",variant],warmup_order=[],
                pair_schedule=[dict(pair=1,order=[arm])],plan_name=case+"x12",
                session=record["round"],phase="full",study="optimization")
            if folder.exists() and json.loads((folder / "run.json").read_text())["status"]=="complete":
                if json.loads((folder / "run.json").read_text())["config"]!=config:
                    raise RuntimeError("Cached full inference configuration differs")
                continue
            run_case(out / "full",name,config)
            print(f"Completed {name}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("preflight","preflight-worker","blocks","full"))
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    parser.add_argument("--phase",choices=("pilot","confirm"),default="pilot")
    parser.add_argument("--sessions",type=int,default=1)
    parser.add_argument("--pairs",type=int,default=2)
    parser.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument("--variant",choices=("organized","batched","reciprocal_only"),default="batched")
    parser.add_argument("--case",choices=tuple(CASES),default="S10R2")
    args = parser.parse_args(); out = args.out.resolve()
    if args.command == "preflight": launch_preflight(out)
    elif args.command == "preflight-worker": correctness_and_compile(out)
    elif args.command=="blocks": run_blocks(out,args.phase,args.sessions,args.pairs,args.variants)
    else: run_full(out,args.sessions,args.variant,args.case)
