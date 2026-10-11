"""Isolated two-party ViT baseline and compact, reproducible measurements."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parent
PAPER = {"f_mmul": (73, 90.91), "f_tensordot": (1, 6.57),
         "f_batch_mmul": (24, 23.81), "_mux": (120, 1.15),
         "f_less": (145, 16.18), "f_exp": (12, 36.18),
         "f_reciprocal": (12, 3.71), "f_mul": (174, 29.73),
         "f_rsqrt": (25, 0.77), "f_gelu": (12, 29.85)}


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    return result.stdout.strip()


def node(out, rank):
    from spu import api, spu_pb2
    import spu.utils.distributed as ppd
    ppd.logger.setLevel(30)
    # Upstream disables rank-1 profiling. Observe both parties without changing
    # any protocol, precision or computation; the native library is untouched.
    original = api.Runtime.__init__

    def observed(self, link, config):
        copied = spu_pb2.RuntimeConfig()
        copied.CopyFrom(config)
        copied.enable_hal_profile = copied.enable_pphlo_profile = True
        original(self, link, copied)

    api.Runtime.__init__ = observed
    Path("/tmp/bb-node.pid").write_text(str(os.getpid()))
    conf = json.loads((out / "run.json").read_text())["spu"]
    ppd.RPC.serve(f"node:{rank}", conf["nodes"])


def worker(out):
    import jax
    import numpy as np
    import spu
    from spu import spu_pb2
    import spu.utils.distributed as ppd
    import spu.utils.frontend as frontend
    ppd.logger.setLevel(30)
    from model import inference, load

    meta = json.loads((out / "run.json").read_text())
    model, pixels = load(Path(meta["model"]), Path(meta["image"]))
    reference = np.asarray(model(pixels, params=model.params, train=False).logits)
    conf = meta["spu"]
    ppd.init(conf["nodes"], conf["devices"])
    # Model and image are owned by different parties, as in the 2PC setting.
    params = ppd.device("P1")(lambda x: x)(model.params)
    inputs = ppd.device("P2")(lambda x: x)(pixels)
    original = frontend.compile
    compilation = {}

    def audited(*args, **kwargs):
        start = time.perf_counter()
        executable, tree = original(*args, **kwargs)
        code = executable.code
        compilation.update(seconds=time.perf_counter() - start,
                           gelu_calls=code.count(b"f_gelu"),
                           sha256=hashlib.sha256(code).hexdigest())
        if compilation["gelu_calls"] != 12:
            raise RuntimeError(f"Expected 12 intrinsic GELUs: {compilation}")
        return executable, tree

    frontend.compile = audited
    copts = spu_pb2.CompilerOptions()
    copts.enable_optimize_denominator_with_broadcast = True
    start = time.perf_counter()
    meta["secure_request_started_at"] = time.time()
    write(out / "run.json", meta)
    print("Starting secure ViT inference", flush=True)
    try:
        secret = ppd.device("SPU")(inference(model), copts=copts)(params, inputs)
        actual = np.asarray(ppd.get(secret))
    finally:
        frontend.compile = original
    elapsed = time.perf_counter() - start
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    result = dict(repeat=meta["active_repeat"], request_s=elapsed,
                  compile_s=compilation["seconds"], gelu_calls=compilation["gelu_calls"],
                  relative_l2=float(np.linalg.norm(diff) / np.linalg.norm(reference)),
                  max_abs=float(np.max(np.abs(diff))), finite=finite,
                  reference_top1=int(reference.argmax()), secure_top1=int(actual.argmax()))
    gate = meta["config"]["gate"]
    result["numerical_pass"] = (finite and result["relative_l2"] <= gate["relative_l2_max"]
                                and result["max_abs"] <= gate["max_abs_max"]
                                and result["reference_top1"] == result["secure_top1"])
    vectors = out / "logits.npz"
    stored = dict(np.load(vectors)) if vectors.exists() else {"reference": reference}
    stored[f"b{result['repeat']}"] = actual
    np.savez_compressed(vectors, **stored)
    meta.setdefault("measurements", []).append(result)
    meta["compilation"] = compilation
    meta["input_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
    import flax
    import transformers
    meta["versions"] = dict(jax=jax.__version__, flax=flax.__version__,
                            transformers=transformers.__version__, spu=getattr(spu, "__version__", None))
    write(out / "run.json", meta)
    print(json.dumps(result), flush=True)


def profile(path):
    text = path.read_text(errors="replace")
    timings = re.findall(r"execution took ([\d.eE+-]+)s", text)
    links = re.findall(r"Link details: total send ([\d.eE+-]+) MiB recv ([\d.eE+-]+) MiB", text)
    operations = {}
    for name, count, seconds, sent, recv in re.findall(
            r"- (\w+), executed (\d+) times, duration ([\d.eE+-]+)s, "
            r"send ([\d.eE+-]+) MiB, recv ([\d.eE+-]+) MiB", text):
        operations[name] = dict(calls=int(count), seconds=float(seconds),
                                sent_mib=float(sent), recv_mib=float(recv))
    if len(timings) != 1 or len(links) != 1:
        raise RuntimeError(f"Incomplete native profile: {path.name}")
    result = dict(seconds=float(timings[0]), sent_mib=float(links[0][0]),
                  recv_mib=float(links[0][1]), operations=operations)
    metrics = re.findall(r"\[BumbleBeeBackend\] ([^\n]+)", text)
    if metrics:
        result["backend_metrics"] = dict(re.findall(r"(\w+)=([\w.eE+-]+)", metrics[-1]))
    return result


def baseline(config_path, requested_out, backend=None):
    config = json.loads(config_path.read_text())
    config["backend"] = backend or config["backend"]
    gpu = config["backend"] == "gpu"
    for key in ("spu_package", "gpu_library", "phantom"):
        config[key] = str((config_path.parent / config[key]).resolve())
    if requested_out:
        out = requested_out.resolve()
    else:
        base = ROOT / "out"
        number = max((int(p.name) for p in base.glob("*") if p.is_dir() and p.name.isdigit()), default=0) + 1
        out = base / f"{number:02}"
    upstream = (config_path.parent / config["upstream"]).resolve()
    workspace = upstream.parent
    model = (config_path.parent / config["model"]).resolve()
    image = (config_path.parent / config["image"]).resolve()
    model.relative_to(workspace)
    image.relative_to(workspace)
    source = config["source_container"]
    merged = command(["docker", "inspect", source, "--format", "{{.GraphDriver.Data.MergedDir}}"])
    original_library = Path(config["spu_package"]) / "spu/libspu.so"
    library = dict(library="/opt/bb/spu/libspu.so", host_file=str(original_library), sha256=digest(original_library))
    if gpu:
        native = Path(config["gpu_library"]).resolve(strict=True)
        library.update(original_cpu_sha256=library["sha256"], host_file=str(native), sha256=digest(native))
    tag = "ours-bb:base"
    prefix = f"ours-bb-{out.name}"
    network = f"{prefix}-net"
    containers = [f"{prefix}-p{rank}" for rank in range(2)]
    threads = config["threads"]
    env = dict(PYTHONPATH=f"{config['dependency_path']}:/opt/bb:/work/OURS/bumblebee",
               JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="0" if gpu else "", HF_HUB_OFFLINE="1",
               SPU_BUMBLEBEE_HE_BACKEND=config["backend"], SPU_BB_ENABLE_MMUL_PACK="1",
               SPU_BB_SET_IEQUAL_BITS="16", OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1",
               YACL_NUM_THREADS=str(threads),
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    if gpu:
        env["LD_LIBRARY_PATH"] = "/opt/openbumblebee-phantom/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
    inside_out = "/work/" + str(out.relative_to(workspace))
    meta = dict(status="preparing", config=config, library=library,
                model="/work/" + str(model.relative_to(workspace)),
                image="/work/" + str(image.relative_to(workspace)),
                hashes={"checkpoint": digest(model / "model.safetensors"), "image": digest(image)},
                workspace_revision=command(["git", "-C", str(upstream), "rev-parse", "HEAD"]),
                backend="existing Phantom CUDA" if gpu else "original CPU SEAL",
                native_provenance="Cached binary identified by SHA256; reference source hashes do not prove build identity",
                precision="FM64/f18; intrinsic GELU FM32/f12; Exp FM64",
                network="separate egress: 1 Gbps and 0.25 ms delay per party",
                timing="three independent fresh SPU runtimes; native execution excludes model loading and compilation",
                environment=env)
    cpu_name = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name"))
    meta["hardware"] = dict(cpu=cpu_name, logical_cpus=os.cpu_count(), party_cpus=config["cpus"],
                            yacl_threads=threads, deployment="two isolated parties on one physical host")
    if gpu:
        meta["hardware"]["gpus"] = [command(["nvidia-smi", "-i", device,
            "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]) for device in config["gpus"]]
        meta["hashes"]["Phantom"] = digest(Path(config["phantom"]) / "lib/libPhantom.so")
    for filename in ("run.py", "runtime.py", "model.py", "config.json"):
        meta["hashes"][filename] = digest(ROOT / filename)
    source_files = ["libspu/mpc/bumblebee/protocol.cc", "libspu/mpc/bumblebee/arith/bumblebee_dot.cc",
                    "libspu/kernel/hal/intrinsic/nn/activation.cc", "libspu/kernel/hal/fxp_approx.cc",
                    "spu/utils/frontend.py", "spu/utils/distributed.py"]
    meta["source_hashes"] = {name: digest(upstream / name) for name in source_files}
    meta["python_spu_hashes"] = {name: digest(Path(config["spu_package"]) / name)
                                for name in ("spu/utils/frontend.py", "spu/utils/distributed.py")}
    out.mkdir(parents=True, exist_ok=False)
    (out / "log").mkdir()
    write(out / "run.json", meta)
    print(f"Baseline: {out.relative_to(ROOT)}", flush=True)
    processes = []
    handles = []
    try:
        exists = subprocess.run(["docker", "image", "inspect", tag], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        if not exists:
            with (out / "log" / "setup.log").open("w") as log:
                subprocess.run(["docker", "build", "--network", "host", "--build-arg",
                                f"BASE={config['image_base']}", "-t", tag, str(ROOT)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        command(["docker", "network", "create", "--internal", network])
        addresses = []
        for rank, name in enumerate(containers):
            args = ["docker", "run", "-d", "--name", name, "--network", network,
                    "--cap-add", "NET_ADMIN", "--cpuset-cpus", config["cpus"][rank],
                    "--memory", "64g", "--shm-size", "1g", "-w", "/work/OURS/bumblebee",
                    "-v", f"{workspace}:/work:ro", "-v", f"{upstream}:/workspace/OpenBumbleBee:ro",
                    "-v", f"{out}:{inside_out}:rw"]
            dependency = config["dependency_path"]
            args += ["-v", f"{merged}{dependency}:{dependency}:ro",
                     "-v", f"{config['spu_package']}:/opt/bb:ro"]
            if gpu:
                args += ["--gpus", f"device={config['gpus'][rank]}",
                         "-v", f"{config['gpu_library']}:{library['library']}:ro",
                         "-v", f"{config['phantom']}:/opt/openbumblebee-phantom:ro"]
            for key, value in env.items():
                args += ["-e", f"{key}={value}"]
            command(args + [tag, "sleep", "infinity"])
            address = command(["docker", "inspect", name, "--format",
                               "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"])
            addresses.append(address)
            command(["docker", "exec", name, "tc", "qdisc", "replace", "dev", "eth0", "root",
                     "netem", "limit", "100000", "delay", "0.25ms", "rate", "1gbit"])
        meta["docker_image"] = command(["docker", "image", "inspect", tag, "--format", "{{.Id}}"])
        meta["ping"] = command(["docker", "exec", containers[0], "ping", "-c", "5", "-i", "0.2", addresses[1]])
        runtime_config = dict(protocol="BUMBLEBEE", field="FM64", fxp_fraction_bits=18,
                              enable_pphlo_profile=True, enable_hal_profile=True,
                              enable_pphlo_trace=False, enable_action_trace=False,
                              fxp_exp_mode=0, fxp_exp_iters=6,
                              experimental_enable_colocated_optimization=True,
                              experimental_disable_mmul_split=True)
        meta["spu"] = dict(nodes={f"node:{r}": f"{ip}:9320" for r, ip in enumerate(addresses)},
                           devices={"SPU": dict(kind="SPU", config=dict(
                               node_ids=["node:0", "node:1"],
                               spu_internal_addrs=[f"{ip}:9330" for ip in addresses],
                               runtime_config=runtime_config)),
                               "P1": dict(kind="PYU", config=dict(node_id="node:0")),
                               "P2": dict(kind="PYU", config=dict(node_id="node:1"))})
        meta["status"] = "running"
        write(out / "run.json", meta)
        rows = []
        for repeat in range(1, config["repeats"] + 1):
            meta = json.loads((out / "run.json").read_text())
            meta["active_repeat"] = repeat
            write(out / "run.json", meta)
            processes, handles = [], []
            for rank, name in enumerate(containers):
                handle = (out / "log" / f"b{repeat}-p{rank}.log").open("w")
                handles.append(handle)
                processes.append(subprocess.Popen(["docker", "exec", name, "python", "run.py", "node",
                                                   "--out", inside_out, "--rank", str(rank)],
                                                  stdout=handle, stderr=subprocess.STDOUT))
            time.sleep(2)
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A node failed to start; see party logs")
            print(f"  b{repeat}: ViT-base 12 blocks, 144 Softmax heads", flush=True)
            with (out / "log" / f"b{repeat}.log").open("w") as log:
                subprocess.run(["docker", "exec", containers[0], "python", "run.py", "worker", "--out", inside_out],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=config["timeout_s"])
            stop_nodes(containers, processes, handles)
            processes, handles = [], []
            meta = json.loads((out / "run.json").read_text())
            parts = [profile(out / "log" / f"b{repeat}-p{rank}.log") for rank in range(2)]
            if gpu and any(p.get("backend_metrics", {}).get("backend") != "gpu" or
                           float(p.get("backend_metrics", {}).get("HE_calls", 0)) <= 0 or
                           float(p.get("backend_metrics", {}).get("H2D_bytes", 0)) <= 0 for p in parts):
                raise RuntimeError("GPU execution was not confirmed by native backend metrics")
            for part in parts:
                for op, calls in {"f_mmul": 73, "f_batch_mmul": 24, "f_tensordot": 1, "f_gelu": 12}.items():
                    if part["operations"].get(op, {}).get("calls") != calls:
                        raise RuntimeError(f"ViT graph mismatch: {op} should execute {calls} times")
            row = meta["measurements"][-1]
            row.update(online_s=max(p["seconds"] for p in parts),
                       sent_gib=parts[0]["sent_mib"] / 1024,
                       recv_gib=parts[0]["recv_mib"] / 1024,
                       total_gib=sum(p["sent_mib"] for p in parts) / 1024)
            for a, b in ((0, 1), (1, 0)):
                if abs(parts[a]["sent_mib"] - parts[b]["recv_mib"]) > 0.01:
                    raise RuntimeError("Party communication totals do not close")
            meta.setdefault("profiles", []).append(parts)
            write(out / "run.json", meta)
            with (out / "ops.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["repeat", "operation", "paper_calls", "calls", "paper_s", "p0_s", "p1_s"])
                for i, pair in enumerate(meta["profiles"], 1):
                    for name, (calls, seconds) in PAPER.items():
                        a, b = (p["operations"][name] for p in pair)
                        writer.writerow([i, name, calls, a["calls"], seconds, a["seconds"], b["seconds"]])
            rows.append(row)
            with (out / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"    {row['online_s']:.2f}s, {row['total_gib']:.2f} GiB, numeric={row['numerical_pass']}", flush=True)
            if not row["numerical_pass"]:
                raise RuntimeError("All-Softmax numerical validation failed")
            if repeat == 1 and row["online_s"] > config["gate"]["paper_seconds"] * config["gate"]["latency_ratio"][1]:
                print("  First run exceeds the time gate; omitting redundant repetitions", flush=True)
                break
        gate = config["gate"]
        latency = statistics.median(row["online_s"] for row in rows)
        communication = statistics.median(row["total_gib"] for row in rows)
        lr = latency / gate["paper_seconds"]
        cr = communication / (gate["paper_sent_gib"] + gate["paper_recv_gib"])
        matched = gate["latency_ratio"][0] <= lr <= gate["latency_ratio"][1] and gate["communication_ratio"][0] <= cr <= gate["communication_ratio"][1]
        meta["comparison"] = dict(median_seconds=latency, median_gib=communication,
                                  latency_ratio=lr, communication_ratio=cr, matched=matched,
                                  completed_repeats=len(rows),
                                  threshold_note="Engineering gate, not a tolerance stated in the paper; CPU/GPU backend and colocated hardware differ")
        meta["status"] = "matched" if matched else "outside_gate"
        meta["research_changes_started"] = False
        write(out / "run.json", meta)
        print(f"Paper gate: {meta['status']} (time {lr:.3f}x, communication {cr:.3f}x)", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        meta = json.loads((out / "run.json").read_text())
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
        meta.update(status=status, error=str(exc) or type(exc).__name__)
        if status in ("timeout", "interrupted") and "secure_request_started_at" in meta:
            meta["unfinished_request_s"] = time.time() - meta["secure_request_started_at"]
        write(out / "run.json", meta)
        raise
    finally:
        stop_nodes(containers, processes, handles)
        for name in containers:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "network", "rm", network], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_nodes(containers, processes, handles):
    for name, process in zip(containers, processes):
        if process.poll() is None:
            subprocess.run(["docker", "exec", name, "python", "-c",
                            "import os,signal; from pathlib import Path; os.kill(int(Path('/tmp/bb-node.pid').read_text()),signal.SIGTERM)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
    for handle in handles:
        handle.close()
