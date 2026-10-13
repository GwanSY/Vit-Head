"""Build the current native source in a private container and cache."""
import json
import os
import shutil
import subprocess
from runtime import ROOT, digest, write


def build(backend="cpu"):
    config = json.loads((ROOT / "config.json").read_text())
    folder = ROOT / "build"
    source = folder / "src"
    upstream = (ROOT / config["upstream"]).resolve()
    if not source.exists():
        shutil.copytree(upstream, source, ignore=shutil.ignore_patterns(
            "bazel-*", "__pycache__", "*.pdf", ".git"))
    cache = folder / "cache"
    if not (folder / "cache.ready").exists():
        print("Copying the existing build cache", flush=True)
        cache.mkdir(exist_ok=True)
        stream = subprocess.Popen(["docker", "exec", config["source_container"], "tar",
                                   "-C", "/tmp/bazel-openbumblebee-container-640", "-cf", "-", "."],
                                  stdout=subprocess.PIPE)
        with (folder / "setup.log").open("w") as log:
            unpack = subprocess.run(["docker", "run", "--rm", "-i", "-v", f"{cache}:/cache",
                                     config["image_base"], "tar", "--warning=no-timestamp",
                                     "-xpf", "-", "-C", "/cache"],
                                    stdin=stream.stdout, stdout=log, stderr=log)
        stream.stdout.close()
        if unpack.returncode or stream.wait():
            raise RuntimeError("Build cache copy failed")
        (folder / "cache.ready").touch()
    bazel = folder / "bazel"
    if not bazel.exists():
        subprocess.run(["docker", "cp", config["source_container"] +
                        ":/tmp/bazel-6.4.0", str(bazel)], check=True)
    targets = ["//spu:libspu.so"]
    if backend == "gpu":
        targets += ["//libspu/mpc/bumblebee/rlwe:phantom_evaluator_test",
                    "//libspu/mpc/bumblebee/rlwe:packlwes_test"]
    args = ["docker", "run", "--rm", "--network", "host", "--name", "ours-bb-build",
            "--cpuset-cpus", "25-47,73-95", "--memory", "64g", "-w", "/workspace/OpenBumbleBee",
            "-v", f"{source}:/workspace/OpenBumbleBee",
            "-v", f"{cache}:/tmp/bazel-openbumblebee-container-640",
            "-v", f"{bazel}:/opt/bazel:ro",
            "-v", f"{folder / 'phantom'}:/opt/openbumblebee-phantom:ro",
            "-v", "/usr/local/cuda-12.3:/usr/local/cuda-12.3:ro",
            "-e", "CUDA_PATH=/usr/local/cuda-12.3",
            "-e", "CUDA_HOME=/usr/local/cuda-12.3",
            "-v", f"{folder / 'tools/cmake/data'}:/opt/cmake:ro",
            "-e", "PATH=/opt/cmake/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-v", f"{folder / 'dist'}:/opt/dist:ro"]
    if backend == "gpu": args += ["--gpus", "device=0"]
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        if key in os.environ: args += ["-e",key]
    args += [
            config["image_base"], "/opt/bazel", "--batch",
            "--output_user_root=/tmp/bazel-openbumblebee-container-640", "build",
            "-c", "opt", "--distdir=/opt/dist", "--jobs=24", "--color=no", "--curses=no"]
    args += (["--config=gpu"] if backend == "gpu" else []) + targets
    metadata = dict(status="building", backend=backend, command=args,
                    sources={p.relative_to(source).as_posix(): digest(p)
                             for p in source.rglob("*") if p.is_file()},
                    phantom_sha256=digest(folder / "phantom/lib/libPhantom.so"))
    manifest = folder / f"{backend}.json"
    write(manifest, metadata)
    print(f"Building current {backend.upper()} library", flush=True)
    with (folder / f"{backend}.log").open("w") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    metadata["status"] = "built" if result.returncode == 0 else "failed"
    metadata["returncode"] = result.returncode
    if result.returncode == 0:
        product = cache / "91451686bbf5ebe79baf6c2b1919d0ed/execroot/spulib/bazel-out/k8-opt/bin"
        products = [(f"{backend}.so", "spu/libspu.so")]
        if backend == "gpu":
            products += [("evaluator", "libspu/mpc/bumblebee/rlwe/phantom_evaluator_test"),
                         ("packing", "libspu/mpc/bumblebee/rlwe/packlwes_test")]
        for name, src in products:
            target = folder / name
            if target.exists() and not (folder / (name + ".old")).exists():
                target.rename(folder / (name + ".old"))
            shutil.copy2(product / src, target)
        metadata["library_sha256"] = digest(folder / f"{backend}.so")
    write(manifest, metadata)
    if result.returncode:
        raise RuntimeError(f"Native build failed; see build/{backend}.log")
    print("Native build complete", flush=True)


if __name__ == "__main__":
    build()
