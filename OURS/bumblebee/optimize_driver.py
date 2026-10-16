"""Serial experiment controller; never overlap benchmark parties across cases."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from runtime import ROOT, digest, write
from optimize_study import run_blocks, run_full
from optimize_analysis import analyze


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out / ".optimization.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root_meta = dict(status="running",stage="confirmation_blocks",started_at=time.time(),pid=os.getpid(),
            objective="Validate eliminable overhead with fixed per-logical-head S/R/L types",
            confirm_variants=["baseline","organized","reciprocal_only"],
            independent_runtimes_per_configuration=8,measured_pairs_per_runtime=2,
            inference_trigger="Any representative Block: paired runtime-mean savings CI > 0 and six-comparison Holm exact p < .05",
            full_rounds_if_triggered=6,
            methodological_limits="Single first-Block activation and fixed image; native CPU SEAL, original FM64/f18 and GELU FM32/f12")
        write(out / "run.json",root_meta)
        source_names=("optimized_heads.py","optimize_study.py","runtime.py","heads.py","block.py",
                      "model.py","experiment.py","run.py","config.json")
        provenance=dict(native_sha256=digest(ROOT / "build/stage2.so"),
            original_cpu_sha256=digest(ROOT / "build/spu/libspu.so"),
            checkpoint_sha256=digest(ROOT / "../../BOLT/checkpoints/vit_b16_hf/model.safetensors"),
            sources={name:digest(ROOT / name) for name in source_names},baseline_revision="d958885b",
            note="No native protocol code or pretrained weights changed; explicit inverse lowers to existing ReciprocalOp")
        write(out / "confirmation_provenance.json",provenance)
        snapshot=out / "provenance/sources"; snapshot.mkdir(parents=True,exist_ok=True)
        for name in source_names: (snapshot / name).write_bytes((ROOT / name).read_bytes())
        try:
            run_blocks(out,"confirm",8,2,["baseline","organized","reciprocal_only"])
            root_meta.update(stage="block_analysis",blocks_completed_at=time.time());write(out / "run.json",root_meta)
            analyze(out)
            result=json.loads((out / "analysis.json").read_text())
            eligible=[r for r in result["summary"] if r["phase"]=="confirm"
                      and r["saved_ci_low_s"]>0 and r["p_holm"]<.05]
            if eligible:
                # Prefer the candidate with highest confirmed paired mean benefit;
                # do not extrapolate predictions as an actual model speedup.
                best=max(eligible,key=lambda r:r["saved_mean_percent"])
                root_meta.update(stage="full_inference",selected_variant=best["variant"],selected_case=best["case"],full_trigger=best)
                write(out / "run.json",root_meta)
                run_full(out,6,best["variant"],best["case"])
                analyze(out)
            else:
                root_meta.update(full_inference_skipped=True,
                    full_skip_reason="No candidate passed the prespecified independent-runtime uncertainty and Holm gate")
            root_meta.update(status="experiments_complete",stage="report",experiments_completed_at=time.time())
            write(out / "run.json",root_meta)
        except BaseException as exc:
            root_meta.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
            write(out / "run.json",root_meta)
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    main(parser.parse_args().out.resolve())
