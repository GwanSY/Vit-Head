"""One complete first ViT Block with shared activations and private weights."""
import copy
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from runtime import ROOT, digest, write
from experiment import case_config, run_case


def block_cases():
    return [dict(case=name,types=types,head_count=12,layout=layout,organization=organization)
        for name,types,layout,organization in (
            ("S12","S"*12,"joint","native"),
            ("SR1","R"+"S"*11,"joint","gather"),
            ("SR6","S"*6+"R"*6,"joint","gather"),
            ("SR10","S"*2+"R"*10,"joint","gather"),
            ("SL6","S"*6+"L"*6,"joint","gather"),
            ("SRL633","S"*6+"R"*3+"L"*3,"joint","gather"),
            ("R12","R"*12,"joint","gather"),
            ("L12","L"*12,"joint","gather"),
            ("SR6-split","S"*6+"R"*6,"split","gather"),
            ("SR6-slices","S"*6+"R"*6,"joint","slices"))]


def block_worker(out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from spu import spu_pb2
    import spu.utils.distributed as ppd
    import spu.utils.frontend as frontend
    from transformers.models.vit.modeling_flax_vit import FlaxViTLayerCollection
    import heads
    from model import load, secure_gelu
    from mixed import sliced_attention
    ppd.logger.setLevel(30)
    meta=json.loads((out / "run.json").read_text());c=meta["config"]
    model,pixels=load(Path(meta["model"]),Path(meta["image"]))
    original=model(pixels,params=model.params,train=False,output_hidden_states=True)
    hidden=np.asarray(original.hidden_states[0],dtype=np.float32)
    params={"0":model.params["vit"]["encoder"]["layer"]["0"]}
    config=copy.deepcopy(model.config);config.num_hidden_layers=1
    layer=FlaxViTLayerCollection(config,dtype=jnp.float32)
    plan=dict(heads=[c["types"]],alpha=[c["alpha"]])
    @contextmanager
    def organization():
        saved=heads.attention
        if c["organization"]=="slices":heads.attention=sliced_attention
        try:
            with heads.bind(plan,c["layout"]):yield
        finally:heads.attention=saved
    def compute(params,hidden,secure=False):
        with organization():
            if secure:
                with secure_gelu():return layer.apply({"params":params},hidden)[0]
            return layer.apply({"params":params},hidden)[0]
    reference=np.asarray(compute(params,jnp.asarray(hidden)))
    if set(c["types"])=={"S"}:
        embedding_error=float(np.max(np.abs(reference-np.asarray(original.hidden_states[1]))))
        if embedding_error>1e-5:raise RuntimeError(f"Standalone Block differs from original first Block: {embedding_error}")
    else:embedding_error=0.
    organization_error=0.
    if c["organization"]=="slices":
        with heads.bind(plan,c["layout"]):
            standard=np.asarray(layer.apply({"params":params},jnp.asarray(hidden))[0])
        organization_error=float(np.max(np.abs(reference-standard)))
        if organization_error>1e-6:raise RuntimeError("Slicing changed first-Block arithmetic")
    ppd.init(meta["spu"]["nodes"],meta["spu"]["devices"]);device=ppd.device("SPU")
    owned_params=ppd.device("P1")(lambda x:x)(params)
    secure_params=jax.tree_util.tree_map(lambda x:ppd.Device.move(x,device),owned_params)
    owned_hidden=ppd.device("P2")(lambda x:x)(hidden)
    # Keep weights private, but share the Block input as a real intermediate.
    original_colocated=device.runtime_config.experimental_enable_colocated_optimization
    device.runtime_config.experimental_enable_colocated_optimization=False
    try:secure_hidden=ppd.Device.move(owned_hidden,device)
    finally:device.runtime_config.experimental_enable_colocated_optimization=original_colocated
    def storage(server,values):
        return [spu_pb2.ValueMetaProto.FromString(x.spu_share.meta).storage_type for x in values]
    flat_params=jax.tree_util.tree_leaves(secure_params)
    input_storage=[client.run_return(storage,[secure_hidden.refs[rank]]) for rank,client in enumerate(device.node_clients)]
    weight_storage=[client.run_return(storage,[x.refs[rank] for x in flat_params]) for rank,client in enumerate(device.node_clients)]
    if any("ashr" not in s.lower() for p in input_storage for s in p):raise RuntimeError("Block input is not shared")
    if any("priv" not in s.lower() for p in weight_storage for s in p):raise RuntimeError("Weights must remain private")
    compilation={};original_compile=frontend.compile
    def audited(*args,**kwargs):
        executable,tree=original_compile(*args,**kwargs)
        compilation.update(gelu_calls=executable.code.count(b"f_gelu"),sha256=hashlib.sha256(executable.code).hexdigest())
        if compilation["gelu_calls"]!=1:raise RuntimeError("Expected one intrinsic GELU per Block")
        return executable,tree
    frontend.compile=audited
    options=spu_pb2.CompilerOptions();options.enable_optimize_denominator_with_broadcast=True
    secure=device(lambda p,x:compute(p,x,True),copts=options)
    try:
        start=time.perf_counter();ppd.get(secure(secure_params,secure_hidden));warmup=time.perf_counter()-start
        for sample in range(1,6):
            start=time.perf_counter();actual=np.asarray(ppd.get(secure(secure_params,secure_hidden)));elapsed=time.perf_counter()-start
            delta=actual.astype(np.float64)-reference.astype(np.float64)
            relative=float(np.linalg.norm(delta)/np.linalg.norm(reference));maximum=float(np.max(np.abs(delta)))
            finite=bool(np.isfinite(actual).all())
            result=dict(repeat=sample,request_s=elapsed,warmup_request_s=warmup,relative_l2=relative,max_abs=maximum,finite=finite,
                numerical_pass=finite and relative<=.05 and maximum<=.5,input_storage_types=input_storage,
                organization_reference_max_abs=organization_error,baseline_block_reference_max_abs=embedding_error)
            meta.setdefault("measurements",[]).append(result);meta["compilation"]=compilation
            meta["block_input_sha256"]=hashlib.sha256(hidden.tobytes()).hexdigest()
            meta["weight_storage_types"]=weight_storage
            meta["block_scope"]="LN1, QKV, attention QK/map/AV, output projection, residual, LN2, FFN with intrinsic GELU, residual"
            write(out / "run.json",meta)
    finally:frontend.compile=original_compile


def run_blocks(out,data,native):
    from mixed import measure
    suite=block_cases();write(out / "block_cases.json",suite)
    alpha=json.loads((data / "alpha.json").read_text())["alpha"][0]
    for index,c in enumerate(suite,1):
        folder=out / "blocks/log" / c["case"]
        config=case_config("block","cpu",study="mixed",types=list(c["types"]),alpha=alpha,
            layout=c["layout"],organization=c["organization"],native_library=str(native),
            trace_intervals=True,repeats=1,measured_calls=5,plan_name=c["case"])
        if folder.exists() and json.loads((folder / "run.json").read_text())["status"]=="complete":
            meta=json.loads((folder / "run.json").read_text())
            if meta["config"]!=config:raise RuntimeError("Cached Block configuration differs")
        else:run_case(out / "blocks",c["case"],config)
        meta=json.loads((folder / "run.json").read_text());meta["_folder"]=str(folder);measure(meta,c["case"])
        print(f"Complete ViT Block: {index}/{len(suite)} configurations ({c['case']})",flush=True)
        status=json.loads((out / "run.json").read_text())
        status.update(stage="blocks",block_completed=index,block_total=len(suite),full_new_runs_authorized=False)
        write(out / "run.json",status)
