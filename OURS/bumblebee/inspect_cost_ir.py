"""Offline compilation audit only; no execution times are benchmark results.

The emitted IR must match executable SHA256 from actual two-party execution
before it is used to associate operation shapes with recorded event intervals.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from runtime import ROOT, command, write


def compile_blocks(out,model,frontend,spu_pb2):
    import copy
    from contextlib import contextmanager
    import jax
    import jax.numpy as jnp
    import numpy as np
    import heads
    from mixed import sliced_attention
    from model import secure_gelu
    from transformers.models.vit.modeling_flax_vit import FlaxViTLayerCollection
    config=copy.deepcopy(model.config);config.num_hidden_layers=1
    layer=FlaxViTLayerCollection(config,dtype=jnp.float32)
    params={"0":model.params["vit"]["encoder"]["layer"]["0"]}
    args=(jax.tree_util.tree_map(lambda x:np.zeros(x.shape,x.dtype),params),np.zeros((1,197,768),np.float32))
    flat,_=jax.tree_util.tree_flatten(args)
    options=spu_pb2.CompilerOptions();options.enable_optimize_denominator_with_broadcast=True
    sources={p.parent.name:p for p in (ROOT / 'out/11/blocks/log').glob('*/run.json') if '-err' not in p.parent.name}
    sources.update({p.parent.name:p for p in (out / 'blocks/log').glob('*/run.json') if '-err' not in p.parent.name})
    destination=out / 'compiler';records=[]
    for name,path in sources.items():
        meta=json.loads(path.read_text());c=meta['config'];plan=dict(heads=[c['types']],alpha=[c['alpha']])
        @contextmanager
        def organization():
            saved=heads.attention
            if c['organization']=='slices':heads.attention=sliced_attention
            try:
                with heads.bind(plan,c['layout']):yield
            finally:heads.attention=saved
        def compute(params,hidden,secure=False):
            with organization():
                if secure:
                    with secure_gelu():return layer.apply({'params':params},hidden)[0]
                return layer.apply({'params':params},hidden)[0]
        fn=lambda p,x:compute(p,x,True)
        executable,_=frontend.compile(frontend.Kind.JAX,fn,args,{},[f'in{i}' for i in range(len(flat))],
            [spu_pb2.Visibility.VIS_SECRET]*len(flat),lambda xs:[f'out{i}' for i in range(len(xs))],copts=options)
        sha=hashlib.sha256(executable.code).hexdigest();matched=sha==meta['compilation']['sha256']
        print('Block',name,sha,'exact_match',matched,flush=True)
        if not matched:raise RuntimeError('Block compiler shape audit does not match actual execution')
        (destination / f'block-{name}.mlir').write_bytes(executable.code)
        records.append(dict(case=name,code_sha256=sha,exact_match=matched,real_metadata=str(path)))
    write(destination / 'block_audit.json',dict(all_exact_matches=True,records=records))


def worker(out,blocks=False):
    import jax
    import numpy as np
    from spu import spu_pb2
    import spu.utils.frontend as frontend
    from model import inference, load
    model,pixels=load(Path("/work/BOLT/checkpoints/vit_b16_hf"),Path("/work/datasets/imagenet2012/val/images/ILSVRC2012_val_00039905.JPEG"))
    if blocks:return compile_blocks(out,model,frontend,spu_pb2)
    design=json.loads((out / "design.json").read_text())
    mock=jax.tree_util.tree_map(lambda x:np.zeros(x.shape,x.dtype),(model.params,pixels))
    flat,_=jax.tree_util.tree_flatten(mock)
    options=spu_pb2.CompilerOptions();options.enable_optimize_denominator_with_broadcast=True
    records=[]
    destination=out / "compiler";destination.mkdir(exist_ok=True)
    for name,plan in design["plans"].items():
        chosen=None if name=="E0" else dict(heads=plan["heads"],alpha=plan["alpha"])
        executable,_=frontend.compile(frontend.Kind.JAX,inference(model,chosen,"joint"),mock,{},
            [f"in{i}" for i in range(len(flat))],[spu_pb2.Visibility.VIS_SECRET]*len(flat),
            lambda xs:[f"out{i}" for i in range(len(xs))],copts=options)
        (destination / f"{name}.mlir").write_bytes(executable.code)
        sha=hashlib.sha256(executable.code).hexdigest()
        actual=[json.loads(p.read_text())["compilation"]["sha256"] for p in (out / "full/log").glob(f"r*-{name}/run.json")
                if json.loads(p.read_text()).get("compilation")]
        matched=bool(actual) and all(v==sha for v in actual)
        records.append(dict(case=name,code_sha256=sha,real_execution_sha256=actual,exact_match=matched,
                            purpose="Static shape/ordering audit only; not secure execution or a timing sample"))
        print(name,sha,"exact_match",matched,flush=True)
        if not matched: raise RuntimeError("Offline compilation does not match real execution")
    write(destination / "audit.json",dict(all_exact_matches=True,records=records))


def launch(out,blocks=False):
    merged=command(["docker","inspect","openbumblebee","--format","{{.GraphDriver.Data.MergedDir}}"])
    inside="/work/"+str(out.relative_to(ROOT.parents[1]))
    args=["docker","run","--rm","--network","none","--name","ours-bb-cost-ir","--cpuset-cpus","25-26","--memory","16g",
          "-w","/work/OURS/bumblebee","-v",f"{ROOT.parents[1]}:/work:ro","-v",f"{out}:{inside}:rw",
          "-v",f"{ROOT}/build:/opt/bb:ro","-v",f"{ROOT}/build/stage2.so:/opt/bb/spu/libspu.so:ro",
          "-v",f"{ROOT}/build/phantom:/opt/openbumblebee-phantom:ro","-v","/usr/local/cuda-12.3:/opt/cuda:ro",
          "-v",f"{merged}/tmp/openbumblebee-pydeps-py310:/tmp/openbumblebee-pydeps-py310:ro"]
    env=dict(PYTHONPATH="/tmp/openbumblebee-pydeps-py310:/opt/bb:/work/OURS/bumblebee",JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",
             HF_HUB_OFFLINE="1",SPU_BUMBLEBEE_HE_BACKEND="cpu",SPU_BB_ENABLE_MMUL_PACK="1",SPU_BB_SET_IEQUAL_BITS="16",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",YACL_NUM_THREADS="1",
             XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
             LD_LIBRARY_PATH="/opt/openbumblebee-phantom/lib:/opt/cuda/lib64")
    for k,v in env.items(): args.extend(("-e",f"{k}={v}"))
    subprocess.run(args+["ours-bb:base","python","inspect_cost_ir.py","--worker","--out",inside]+(['--blocks'] if blocks else []),check=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/12")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--blocks",action="store_true")
    args=parser.parse_args()
    (worker if args.worker else launch)(args.out.resolve(),args.blocks)
