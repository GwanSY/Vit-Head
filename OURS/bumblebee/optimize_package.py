"""Freeze reproducible source, tables and all raw paired execution records."""
import argparse
import csv
import hashlib
import json
import tarfile
import time
import zipfile
from pathlib import Path

from runtime import ROOT, digest, write
from experiment import save_rows


def package(out):
    meta=json.loads((out / "run.json").read_text())
    if meta["status"] not in ("experiments_complete","complete"):
        raise RuntimeError("Finish the experiments and numerical checks before packaging")
    analysis=json.loads((out / "analysis.json").read_text())
    if analysis["partial"] or not analysis["all_numerical_pass"] or not analysis["all_closure_pass"]:
        raise RuntimeError("Incomplete numerical or actual-execution evidence")
    identity=json.loads((out / "confirmation_provenance.json").read_text())
    for name,sha in identity["sources"].items():
        if digest(ROOT / name)!=sha: raise RuntimeError(f"Measured computation changed: {name}")
    native=ROOT / "build/stage2.so"
    if digest(native)!=identity["native_sha256"]: raise RuntimeError("Actual native library changed")
    sources=out / "provenance/sources"; sources.mkdir(parents=True,exist_ok=True)
    for name in ("optimized_heads.py","optimize_study.py","optimize_driver.py","optimize_analysis.py",
        "optimize_report.py","optimize_package.py","runtime.py","heads.py","block.py","model.py",
        "experiment.py","run.py","config.json","mixed.py","group_costs.py","cost_analysis.py",
        "profile_build.py","build.py","Dockerfile"):
        (sources / name).write_bytes((ROOT / name).read_bytes())
    for name in ("stage2.json","stage2.diff"):
        original=ROOT / "build" / name
        if original.exists(): (out / "provenance" / name).write_bytes(original.read_bytes())
    historical=out / "historical"; historical.mkdir(exist_ok=True)
    for name in ("summary.txt","cost_summary.csv","group_hlo_summary.csv","protocol_audit.json",
                 "full_component_contrasts.csv","paired_full_contrasts.csv","audit.json"):
        original=ROOT / "out/12" / name
        if original.exists(): (historical / name).write_bytes(original.read_bytes())
    graph_checks=[]
    paths={"S10R2":"block-S10R2.mlir","S6R6":"block-SR6.mlir","S6R3L3":"block-SRL633.mlir"}
    for case,old_name in paths.items():
        old=ROOT / "out/12/compiler" / old_name
        new=out / "compiler" / f"{case}-baseline.mlir"
        identical=old.read_text().splitlines()[1:]==new.read_text().splitlines()[1:]
        if not identical: raise RuntimeError(f"Baseline graph changed: {case}")
        graph_checks.append(dict(case=case,historical_sha256=digest(old),current_sha256=digest(new),
            identical_after_module_symbol=True))
    old_slices=ROOT / "out/12/compiler/block-SR6-slices.mlir"
    new_slices=out / "compiler/S6R6-organized.mlir"
    if old_slices.read_text().splitlines()[1:]!=new_slices.read_text().splitlines()[1:]:
        raise RuntimeError("Existing slices control changed")
    write(out / "baseline_graph_reuse.json",dict(baselines=graph_checks,organized_matches_old_slices=True,
        caveat="The public organization path is an existing control; its new evidence is paired measurement, not a new cryptographic algorithm"))
    raw=[]
    for phase in ("pilot","confirm","full","aborted"):
        folder=out / phase
        if folder.exists():
            raw+=sorted(p for p in folder.rglob("*") if p.is_file())
    write(out / "execution_completion_frozen.json",meta)
    raw += [out / "confirmation_provenance.json",out / "preflight.json",out / "execution_completion_frozen.json",out / "archived_attempts.json"]
    raw += list((out / "compiler").glob("*.mlir"))
    raw += list(out.glob("*_design.json"))
    raw = sorted(set(p for p in raw if p.exists()))
    manifest=[dict(archive_path="raw/"+str(p.relative_to(out)),original_path=str(p),
                   bytes=p.stat().st_size,sha256=digest(p)) for p in raw]
    manifest.append(dict(archive_path="native/stage2.so",original_path=str(native),
                         bytes=native.stat().st_size,sha256=digest(native)))
    save_rows(out / "raw_manifest.csv",manifest)
    archive=out / "raw_executions.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for row in manifest: tar.add(row["original_path"],arcname=row["archive_path"],recursive=False)
        tar.add(out / "raw_manifest.csv",arcname="raw_manifest.csv",recursive=False)
        tar.add(out / "REPRODUCE.txt",arcname="REPRODUCE.txt",recursive=False)
    # Stream verification against the manifest, not just checking archive existence.
    expected={r["archive_path"]:r for r in manifest};verified=0
    with tarfile.open(archive,"r|gz") as tar:
        for member in tar:
            if member.name not in expected: continue
            row=expected[member.name];h=hashlib.sha256();size=0
            stream=tar.extractfile(member)
            for chunk in iter(lambda:stream.read(1<<20),b""): h.update(chunk);size+=len(chunk)
            if size!=row["bytes"] or h.hexdigest()!=row["sha256"]: raise RuntimeError(f"Raw archive integrity failed: {member.name}")
            verified+=1
    if verified!=len(manifest): raise RuntimeError("Raw archive lacks manifest members")
    zip_path=out / "results.zip"
    selected=list(out.glob("*.csv"))+list(out.glob("*.pdf"))+list(out.glob("*.png"))
    selected += [out / name for name in ("report.html","summary.txt","REPRODUCE.txt","analysis.json",
        "confirmation_provenance.json","preflight.json","baseline_graph_reuse.json","figure_manifest.json",
        "confirm_design.json","full_design.json","archived_attempts.json") if (out/name).exists()]
    selected += list((out / "provenance").rglob("*"))+list(historical.glob("*"))
    selected += list((out / "compiler").glob("*.mlir"))
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive_zip:
        for path in sorted(set(p for p in selected if p.is_file())): archive_zip.write(path,str(path.relative_to(out)))
    with zipfile.ZipFile(zip_path) as archive_zip:
        if archive_zip.testzip() is not None: raise RuntimeError("Results ZIP CRC failure")
    write(out / "package_audit.json",dict(raw_members_verified=verified,all_member_sha256_pass=True,
        raw_archive_sha256=digest(archive),results_zip_sha256=digest(zip_path),results_zip_crc_pass=True,
        measured_calls=analysis["measured_calls"],closure_checks=analysis["closure_checks"],
        native_library_included=True))
    meta.update(status="complete",stage="complete",completed_at=time.time(),
                deliverables="report.html, summary.txt, CSVs, vector PDF/PNG, results.zip, raw_executions.tar.gz")
    write(out / "run.json",meta)
    print(f"Packaged and SHA256-verified {verified} raw members",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    package(parser.parse_args().out.resolve())
