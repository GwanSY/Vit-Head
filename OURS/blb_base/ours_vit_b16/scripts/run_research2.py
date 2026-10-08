#!/usr/bin/env python3
"""Shared configuration entry point; no model or CUDA imports before launch."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from version_matrix import MATRIX, require_version, plan_binding, file_hash
from experiment_contract import CONTRACT, CONTRACT_SHA256, validate_assignment_contract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "ours_vit_b16/experiments/experiment.json"


def rooted(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (ROOT / path).resolve())


def read_config_json(path: Path) -> dict:
    """Accept // comments outside JSON strings; keep newlines for error locations."""
    source = path.read_text(encoding="utf-8")
    source = re.sub(r'"(?:\\.|[^"\\])*"|//[^\r\n]*',
                    lambda match: " " * len(match[0]) if match[0].startswith("//") else match[0],
                    source)
    return json.loads(source)


def load_config(path: Path) -> dict:
    defaults = read_config_json(DEFAULT_CONFIG)
    config = copy.deepcopy(defaults)
    supplied = read_config_json(path)
    for section, values in supplied.items():
        if section not in defaults or not isinstance(values, dict):
            raise ValueError(f"unknown or invalid config section: {section}")
        unknown = values.keys() - defaults[section].keys()
        if unknown:
            raise ValueError(f"unknown {section} options: {sorted(unknown)}")
        config[section].update(values)
    budgets = config["accuracy"]["budgets_pp"]
    if (not isinstance(budgets, list) or not budgets or
            any(type(b) not in (int, float) or not math.isfinite(b) or b < 0
                for b in budgets) or len(set(budgets)) != len(budgets)):
        raise ValueError("accuracy.budgets_pp must be a nonempty list of unique finite nonnegative numbers")
    accuracy = config["accuracy"]
    if accuracy["comparison_budget_pp"] is None:
        accuracy["comparison_budget_pp"] = max(budgets)
    if accuracy["comparison_budget_pp"] not in budgets:
        raise ValueError("accuracy.comparison_budget_pp must be in budgets_pp")
    data = config["data"]
    if data["heldout_start"] is None:
        data["heldout_start"] = data["calibration_start"] + data["calibration_count"]
    secure = config["secure"]
    if secure["backend"] not in {"cpu", "gpu"}:
        raise ValueError("secure.backend must be cpu or gpu")
    if type(secure["gpu_id"]) is not int or secure["gpu_id"] < 0:
        raise ValueError("secure.gpu_id must be a nonnegative physical GPU index")
    pair = secure["gpu_pair"]
    if pair is not None and (not isinstance(pair, list) or len(pair) != 2 or
                             any(type(g) is not int or g < 0 for g in pair)):
        raise ValueError("secure.gpu_pair must be null or [cloud_gpu, edge_gpu]")
    require_version(secure["version"])
    if secure["network_mode"] not in {"netem", "none"}:
        raise ValueError("secure.network_mode must be netem or none")
    for key, upper in (("min_free_memory_mib", None), ("max_gpu_utilization_percent", 100)):
        value = secure[key]
        if type(value) is not int or value < 0 or (upper is not None and value > upper):
            raise ValueError(f"invalid secure.{key}: {value}")
    for key, value in config["paths"].items():
        if value is not None:
            config["paths"][key] = rooted(value)
    for key in ("input", "assignment", "head_plan", "paired_config", "binary_override", "phantom_library_override"):
        value = config["secure"][key]
        if value is not None:
            config["secure"][key] = rooted(value)
    if config["secure"]["output_dir"] is None:
        parent = Path(config["paths"]["output_dir"]) / "secure"
        index = 1
        while (parent / f"run_{index:03d}").exists():
            index += 1
        config["secure"]["output_dir"] = os.path.relpath(
            parent / f"run_{index:03d}", ROOT)
    else:
        config["secure"]["output_dir"] = os.path.relpath(rooted(config["secure"]["output_dir"]), ROOT)
    return config


def offline_command(config: dict) -> list[str]:
    missing = [key for key, value in config["paths"].items() if value is None]
    if missing:
        raise ValueError(f"set paths before running offline selection: {', '.join(missing)}")
    if config['secure']['version'] not in ('HA', 'J'):
        raise ValueError('offline search is only allowed for HA or J; HAP/HP must reuse their source plan')
    options = dict(config["paths"])
    options['version'] = config['secure']['version']
    options["output_dir"] = str(Path(options["output_dir"]) / "offline")
    for section in ("data", "search", "cost_model", "comparisons"):
        options.update(config[section])
    options["budgets_pp"] = config["accuracy"]["budgets_pp"]
    options["final_budget_pp"] = config["accuracy"]["comparison_budget_pp"]
    options["oracle_heads"] = ",".join(map(str, options["oracle_heads"]))
    command = [sys.executable, "-B", str(Path(__file__).with_name("offline_allocator_experiment.py"))]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.append(flag)
            command.extend(map(str, value if isinstance(value, list) else [value]))
    return command


def secure_environment(config: dict) -> dict[str, str]:
    secure = config["secure"]
    mapping = {
        "backend": "OURS_VIT_BACKEND",
        "network_mode": "OURS_VIT_NETWORK_MODE",
        "min_free_memory_mib": "OURS_VIT_MIN_FREE_MEMORY_MIB",
        "max_gpu_utilization_percent": "OURS_VIT_MAX_GPU_UTILIZATION_PERCENT",
        "image": "OURS_DOCKER_IMAGE",
        "binary_override": "OURS_VIT_BINARY_OVERRIDE",
        "phantom_library_override": "OURS_VIT_PHANTOM_LIBRARY_OVERRIDE",
        "input": "OURS_VIT_INPUT",
        "gpu_direct_plaintext": "BLB_VIT_GPU_DIRECT_PLAINTEXT",
        "gpu_linear_streams": "BLB_VIT_GPU_LINEAR_STREAMS",
        "omp_threads": "BLB_VIT_OMP_THREADS",
        "port": "OURS_VIT_PORT",
        "timeout_seconds": "BLB_VIT_TIMEOUT_SECONDS",
    }
    environment = {name: str(secure[key]) if secure[key] is not None else ""
                   for key, name in mapping.items()}
    pair = secure["gpu_pair"] or [secure["gpu_id"], secure["gpu_id"]]
    environment.update(OURS_VIT_PHYSICAL_GPUS=','.join(map(str, pair)) if secure["backend"] == "gpu" else "",
                       OURS_VIT_ALLOW_SHARED_GPU="1" if pair[0] == pair[1] else "0",
                       OURS_VIT_MODEL=config["paths"]["model"])
    return environment


def secure_command(config: dict) -> tuple[list[str], dict[str, str]]:
    secure = config["secure"]
    version = require_version(secure['version'])
    plan = secure['head_plan']
    if secure['assignment']:
        document = json.loads(Path(secure['assignment']).read_text())
        runtime = document['runtime']
        owner = require_version(runtime['version'])
        if owner != version and MATRIX[version]['paired_with'] != owner:
            raise ValueError('assignment version does not match the requested experiment')
        validate_assignment_contract(document)
        plan = runtime['head_plan']
        if document.get('model_sha256') != file_hash(config['paths']['model']):
            raise ValueError('assignment checkpoint hash mismatch')
    if version in ('B', 'BP'):
        if plan or secure['assignment']:
            raise ValueError('B/BP use the built-in all-Softmax plan')
        plan = str(ROOT / 'ours_vit_b16/experiments/all_softmax_head_plan.txt')
    elif not plan:
        raise ValueError('heterogeneous versions require a frozen head plan/assignment')
    plan = rooted(plan)
    binding = plan_binding(plan)
    if secure['assignment'] and document['runtime'].get('semantic_plan_sha256') != binding['semantic_plan_sha256']:
        raise ValueError('assignment frozen semantic plan mismatch')
    if version in ('HP', 'HAP'):
        if not secure['paired_config']:
            raise ValueError('HP/HAP require paired_config pointing to the source run config.secure.json')
        parent = load_config(Path(secure['paired_config']))
        frozen = json.loads(Path(secure['paired_config']).with_name('plan_binding.json').read_text())
        if frozen.get('experiment_contract_sha256') != CONTRACT_SHA256 or frozen.get('experiment_contract') != CONTRACT:
            raise ValueError('paired source experiment definitions missing or incompatible')
        if frozen['semantic_plan_sha256'] != binding['semantic_plan_sha256']:
            raise ValueError('paired source plan was changed after the source run')
        for key, old_hash in frozen['files'].items():
            section, field = key.split('.')
            if file_hash(config[section][field]) != old_hash:
                raise ValueError(f'paired frozen asset changed: {key}')
        if parent['secure']['version'] != MATRIX[version]['paired_with']:
            raise ValueError('wrong paired source version')
        parent_command, _ = secure_command(parent)
        if plan_binding(parent_command[3])['semantic_plan_sha256'] != binding['semantic_plan_sha256']:
            raise ValueError('paired experiment changes head identity/protocol/alpha')
        for key in ('model', 'model_manifest', 'preprocess_manifest'):
            if file_hash(config['paths'][key]) != file_hash(parent['paths'][key]):
                raise ValueError(f'paired experiment changes {key}')
        for key in ('backend', 'gpu_id', 'gpu_pair', 'image', 'binary_override',
                    'phantom_library_override', 'input', 'gpu_direct_plaintext',
                    'gpu_linear_streams', 'omp_threads', 'network_mode'):
            if secure[key] != parent['secure'][key]:
                raise ValueError(f'paired experiment changes secure.{key}')
    config['secure']['head_plan'] = None if version in ('B', 'BP') else plan
    extra = secure_environment(config)
    if version in ('HP', 'HAP'):
        if not frozen.get('image_id', '').startswith('sha256:'):
            raise ValueError('paired source lacks immutable image ID; finish the source run first')
        extra['OURS_DOCKER_IMAGE'] = frozen['image_id']
    output = Path(rooted(secure['output_dir']))
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('secure.output_dir must be below outputs/')
    command = ["bash", str(ROOT / "scripts/run_ours_container.sh"),
               version, plan,
               str(output.relative_to(ROOT)), "12", "full"]
    return command, extra


def edge_cloud_metadata_agrees(metadata: list[dict]) -> bool:
    if len(metadata) != 2:
        return False
    cloud, edge = metadata
    digest = cloud.get('edge_cloud_contract_sha256', '')
    return bool(re.fullmatch(r'[0-9a-f]{64}', digest) and
                edge.get('edge_cloud_contract_sha256') == digest and
                cloud.get('edge_cloud_role') == 'cloud' and
                edge.get('edge_cloud_role') == 'edge' and
                all(m.get('edge_cloud_plan_agreed') == '1' for m in metadata))


def validate_secure_run(config: dict, command: list[str], environment: dict, log) -> int:
    native = Path(rooted(config['secure']['output_dir']))
    reference_path = native / 'reference.json'
    reference = [sys.executable, '-B', str(Path(__file__).with_name('vit_reference.py')),
                 '--model', config['paths']['model'], '--input', config['secure']['input'],
                 '--secure-logits', str(native / 'secure_logits_party2.txt'),
                 '--output', str(reference_path)]
    if command[2] != 'B':
        reference += ['--head-plan', command[3]]
    reference_code = subprocess.call(reference, env=environment, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(reference_path.read_text())
    fixed, actual = report['fixed']['logits'], report['secure']['logits']
    if len(fixed) != len(actual) or not actual or any(not math.isfinite(v) for v in fixed + actual):
        raise ValueError('invalid final logits in numerical audit')
    relative_l2 = math.sqrt(sum((a-b)**2 for a, b in zip(fixed, actual)) /
                            max(sum(v*v for v in fixed), 1e-30))
    maximum = max(abs(a-b) for a, b in zip(fixed, actual))
    totals, metadata = [], []
    for party in (1, 2):
        with (native / f'execution_totals_party{party}.csv').open() as source:
            totals.append({r['metric']: float(r['value']) for r in csv.DictReader(source)})
        metadata.append(dict(line.split('=', 1) for line in
                             (native / f'run_metadata_party{party}.txt').read_text().splitlines() if '=' in line))
    online = max(t['secure_end_to_end_s'] for t in totals)
    same_logits = ((native / 'secure_logits_party1.txt').read_bytes() ==
                   (native / 'secure_logits_party2.txt').read_bytes())
    numerical_pass = bool(reference_code == 0 and same_logits and report['secure']['top1_matches_fixed']
                          and report['secure']['top1'] == report['clear']['top1']
                          and relative_l2 <= .05 and maximum <= .5)
    expected_packing = ('not_applicable' if command[2] == 'B' else
                        'packed' if MATRIX[command[2]]['packing'] else 'unpacked')
    native_pass = all(m.get('version') == command[2] and
                      m.get('attention_packing') == expected_packing and
                      m.get('execution_framework') == 'ours_native_edge_cloud_v1' and
                      m.get('conversion_path') == 'wide_shared_rns_v1' and
                      m.get('attention_diagonal_kernel') == 'complete_v1' and
                      m.get('softmax_product') == 'ckks' and
                      m.get('gelu_polynomial') == 'blb_F0_F1' and
                      m.get('secure_exp_clamp') == '1' and
                      m.get('executed_transformer_blocks') == '12' and m.get('partial_run') == '0'
                      for m in metadata)
    native_pass = native_pass and edge_cloud_metadata_agrees(metadata)
    communication = {
        name: {'party_bytes': [int(t[name]) for t in totals],
               'aggregate_bytes': sum(int(t[name]) for t in totals)}
        for name in ('online_communication_bytes', 'initialization_communication_bytes',
                     'total_communication_bytes')}
    result = dict(execution_pass=native_pass, numerical_pass=numerical_pass, online_s=online,
                  acceptance_pass=native_pass and numerical_pass,
                  version=command[2], new_heterogeneous_packing=MATRIX[command[2]]["packing"],
                  packing_scope="new_matrix_P_only_not_BLB_ciphertext_packing",
                  blb_ciphertext_packing="active_QKV_BSGS_head_pairs",
                  attention_diagonal_kernel=metadata[0].get('attention_diagonal_kernel'),
                  performance_status="MEASURED_REQUIRES_PAIRED_COMPARISON",
                  secure_top1=report['secure']['top1'], fixed_top1=report['fixed']['top1'],
                  clear_top1=report['clear']['top1'], fixed_relative_l2=relative_l2,
                  fixed_max_abs=maximum, same_public_logits=same_logits, totals=totals,
                  communication=communication, communication_scope='sum_of_party_sent_bytes',
                  dataset_accuracy_claimed=False)
    (native / 'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), file=log, flush=True)
    return 0 if numerical_pass and native_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version", choices=tuple(MATRIX), help="唯一实验版本")
    parser.add_argument("--stage", choices=("offline", "costs", "secure"), default="offline")
    parser.add_argument("--show-config", action="store_true", help="print resolved config; do not run anything")
    parser.add_argument("--print-command", action="store_true", help="print command; do not run anything")
    parser.add_argument('--gpus', type=int, nargs=2, metavar=('CLOUD', 'EDGE'),
                        help='physical GPUs for this single secure run; never auto-select other cards')
    parser.add_argument('--output-dir', type=Path, help='output root for this single secure run')
    args = parser.parse_args()
    if (args.gpus or args.output_dir) and args.stage != 'secure':
        parser.error('--gpus/--output-dir apply only to --stage secure')
    if args.gpus and any(g < 0 for g in args.gpus):
        parser.error('GPU indices must be nonnegative')
    config = load_config(args.config)
    if args.version:
        config["secure"]["version"] = args.version
    if args.gpus:
        config['secure']['gpu_pair'] = args.gpus
    if args.output_dir:
        config['paths']['output_dir'] = rooted(str(args.output_dir))
        config['secure']['output_dir'] = str(Path(config['paths']['output_dir']) / 'native')
    if args.show_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    # Remove inherited experiment overrides so this file determines the run.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("OURS_VIT_", "BLB_VIT_"))}
    environment["CUDA_VISIBLE_DEVICES"] = str(config["secure"]["gpu_id"]) if config["secure"]["backend"] == "gpu" else ""
    environment["PYTHONUNBUFFERED"] = "1"
    extra = {}
    if args.stage == "offline":
        command = offline_command(config)
    elif args.stage == "costs":
        command = [sys.executable, "-B", str(Path(__file__).with_name("research2_costs.py")),
                   "--config", str(args.config.resolve()), "--version", config["secure"]["version"]]
        extra = secure_environment(config)
    else:
        command, extra = secure_command(config)
    environment.update(extra)
    if args.stage == 'secure':
        environment['CUDA_VISIBLE_DEVICES'] = extra['OURS_VIT_PHYSICAL_GPUS']
    if args.print_command:
        print(shlex.join(["env", "CUDA_VISIBLE_DEVICES=" + environment["CUDA_VISIBLE_DEVICES"]] +
                         [f"{key}={value}" for key, value in extra.items()] + command))
        return
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / f"config.{args.stage}.json"
    if args.stage == "secure" and snapshot.exists():
        raise ValueError("secure output root already contains a frozen run; choose a new --output-dir")
    snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    if args.stage == 'secure':
        binding = plan_binding(command[3])
        binding.update(version=config['secure']['version'], model_sha256=file_hash(config['paths']['model']))
        binding['experiment_contract'] = CONTRACT
        binding['experiment_contract_sha256'] = CONTRACT_SHA256
        binding['files'] = {f'{section}.{key}': file_hash(config[section][key])
                            for section, keys in (('paths', ('model', 'model_manifest', 'preprocess_manifest')),
                                                  ('secure', ('input', 'binary_override', 'phantom_library_override')))
                            for key in keys if config[section][key]}
        (output_dir / 'plan_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    log_path = output_dir / "run.log"
    started = datetime.now(timezone.utc).isoformat()
    print(f"Research2 {args.stage}: started; log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] START stage={args.stage}\n")
        log.write("config=" + json.dumps(config, ensure_ascii=False) + "\n")
        log.write("command=" + shlex.join(command) + "\n")
        log.flush()
        try:
            code = subprocess.call(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
            if code == 0 and args.stage == 'secure':
                selection = Path(rooted(config['secure']['output_dir'])) / 'gpu_selection.txt'
                image_ids = [line.split('=', 1)[1] for line in selection.read_text().splitlines()
                             if line.startswith('image_id=')]
                if not image_ids:
                    raise ValueError('run lacks immutable Docker image provenance')
                binding['image_id'] = image_ids[-1]
                (output_dir / 'plan_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
                code = validate_secure_run(config, command, environment, log)
        except KeyboardInterrupt:
            code = 130
        except (OSError, ValueError, KeyError) as error:
            log.write(f"Run or validation failed: {error}\n")
            code = 1
        log.write(f"[{datetime.now(timezone.utc).isoformat()}] END stage={args.stage} exit_code={code}\n")
    (output_dir / "last_run.json").write_text(json.dumps({
        "stage": args.stage, "started": started, "exit_code": code,
        "log": str(log_path), "config": str(snapshot),
    }, indent=2) + "\n")
    status = 'completed' if code == 0 else 'failed'
    if args.stage in {"costs", "secure"} and code == 75:
        status = f'blocked: GPU {config["secure"]["gpu_id"]} busy or insufficient free memory; rerun when available'
    if args.stage == "offline" and code == 3:
        status = 'blocked: cost model prerequisites not met'
        report = json.loads((output_dir / "offline/cost_model_gate_report.json").read_text())
        gate = report["cost_model_gate"]
        status += f" ({gate['cost_model_gate_reason']})"
        if gate.get("missing_train_compositions") or gate.get("missing_heldout_compositions"):
            status += '; collect missing measurements with --stage costs, then rerun --stage offline'
    print(f"Research2 {args.stage}: {status} (exit={code}); details={log_path}")
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Research2 configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)
