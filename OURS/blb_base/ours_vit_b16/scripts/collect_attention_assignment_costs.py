#!/usr/bin/env python3
"""Collect directly measured complete A3/A4 attention assignments.

One output row is one whole layer assignment from QKV/QK through the genuine
Algorithm-1 attention-output conversion.  Rows are never divided by Head
count, copied to another layer, or assembled from normalization-only runs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROFILE_PREFIX = "OURS_A3_A4_ASSIGNMENT_PROFILE "
OUTPUT_COLUMNS = (
    "layer", "ablation", "softmax_heads", "relu_heads", "linear_heads",
    "latency_ms", "he_instrumented_ms", "conversion_instrumented_ms",
    "communication_bytes", "rounds", "protocol_groups", "mpc_group_calls",
    "native_linear_groups", "ckks_to_mpc", "mpc_to_ckks",
    "he_ct_pt_multiply", "he_ct_ct_multiply", "he_rotations", "he_rescales",
    "measurement_scope", "cost_unit", "network_gbps", "rtt_ms", "backend",
    "gpu_topology", "image_id", "binary_sha256", "plan_sha256",
    "protocols_original", "alphas_original", "physical_protocols",
    "original_order", "a4_data_level_extension",
    "blb_fast_attention_packing", "attention_packing",
    "heterogeneous_head_grouping", "surrogate", "dynamic_selection",
    "source_run",
)


def key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in text.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


@dataclass(frozen=True)
class Profile:
    values: dict[str, str]

    @property
    def key(self) -> tuple[int, str, int, int, int]:
        return (
            int(self.values["layer"]), self.values["ablation"].upper(),
            int(self.values["softmax_heads"]),
            int(self.values["relu_heads"]),
            int(self.values["linear_heads"]),
        )


def require_run_provenance(run: Path) -> tuple[str, str, str, str, str]:
    status = key_values((run / "status.txt").read_text(encoding="utf-8"))
    if status.get("alice_rc") != "0" or status.get("bob_rc") != "0":
        raise ValueError(f"secure run failed: {run}")
    environment_text = (run / "environment.txt").read_text(encoding="utf-8")
    environment = key_values(environment_text)
    if environment.get("backend") != "gpu":
        raise ValueError(f"assignment cost is not a GPU run: {run}")
    if "network=1Gbps/0.3ms_RTT" not in environment_text:
        raise ValueError(f"assignment cost lacks formal network shaping: {run}")
    ablation = environment.get("ablation", "").upper()
    if ablation not in {"A3", "A4"}:
        raise ValueError(f"assignment cost must be A3 or A4: {run}")
    binary_sha256 = environment.get("binary_sha256", "")
    if len(binary_sha256) != 64:
        raise ValueError(f"assignment cost lacks binary SHA-256: {run}")

    selection = key_values(
        (run / "gpu_selection.txt").read_text(encoding="utf-8"))
    topology = selection.get("party_topology", "")
    if topology not in {"separate_physical_gpus", "shared_physical_gpu"}:
        raise ValueError(f"assignment cost has unknown GPU topology: {run}")
    physical = selection.get("physical_gpus", "").split(",")
    if len(physical) != 2:
        raise ValueError(f"assignment cost has invalid GPU topology: {run}")
    if topology == "separate_physical_gpus" and physical[0] == physical[1]:
        raise ValueError(f"separate-GPU assignment has identical devices: {run}")
    if topology == "shared_physical_gpu" and physical[0] != physical[1]:
        raise ValueError(f"shared-GPU assignment has different devices: {run}")
    image_id = selection.get("image_id", "")
    plan_sha256 = selection.get("plan_sha256", "")
    if not image_id.startswith("sha256:") or len(plan_sha256) != 64:
        raise ValueError(f"assignment cost lacks immutable provenance: {run}")
    return ablation, binary_sha256, image_id, plan_sha256, topology


def read_party_profiles(run: Path, party_name: str,
                        ablation: str) -> dict[tuple[int, str, int, int, int], Profile]:
    log = (run / party_name).read_text(encoding="utf-8", errors="replace")
    if "surrogate=1" in log or "mpc_reference=1" in log or \
            "client_decrypt_reencrypt=1" in log:
        raise ValueError(f"forbidden A4 shortcut appears in {run}/{party_name}")
    profiles: dict[tuple[int, str, int, int, int], Profile] = {}
    for line in log.splitlines():
        if not line.startswith(PROFILE_PREFIX):
            continue
        values = key_values(line)
        required = {
            "status", "ablation", "layer", "latency_ms",
            "communication_bytes", "rounds", "he_instrumented_ms",
            "conversion_instrumented_ms", "softmax_heads", "relu_heads",
            "linear_heads", "protocol_groups", "mpc_group_calls",
            "native_linear_groups", "ckks_to_mpc", "mpc_to_ckks",
            "he_ct_pt_multiply", "he_ct_ct_multiply", "he_rotations",
            "he_rescales", "measurement_scope", "cost_unit", "network",
            "dynamic_selection", "surrogate",
        }
        if required - values.keys():
            raise ValueError(f"incomplete assignment profile in {run}/{party_name}")
        if (values["status"] != "PASS" or
                values["ablation"].upper() != ablation or
                values["measurement_scope"] != "complete_attention_assignment" or
                values["cost_unit"] != "direct_assignment_not_per_head" or
                values["network"] != "1Gbps_0.3ms_RTT" or
                values["dynamic_selection"] != "0" or
                values["surrogate"] != "0"):
            raise ValueError(f"ineligible assignment profile in {run}/{party_name}")
        profile = Profile(values)
        if sum(profile.key[2:]) != 12:
            raise ValueError(f"assignment does not contain 12 Heads: {run}")
        if profile.key in profiles:
            raise ValueError(f"duplicate assignment profile {profile.key}: {run}")
        profiles[profile.key] = profile
    if not profiles:
        raise ValueError(f"no complete assignment profiles in {run}/{party_name}")
    return profiles


def marker_values(log: str, prefix: str) -> list[dict[str, str]]:
    return [key_values(line) for line in log.splitlines()
            if line.startswith(prefix)]


def require_native_markers(run: Path,
                           profiles: dict[tuple[int, str, int, int, int], Profile]) -> None:
    logs = {
        party_name: (run / party_name).read_text(
            encoding="utf-8", errors="replace")
        for party_name in ("alice.log", "bob.log")
    }
    for party_name, log in logs.items():
        chain = marker_values(log, "OURS_A4_HE1_CHAIN ")
        if len(chain) != 1:
            raise ValueError(f"missing unique A4 chain marker: {run}/{party_name}")
        values = chain[0]
        expected_extension = "1" if any(key[4] for key in profiles) else "0"
        if (values.get("status") != "PASS" or
                values.get("enabled") != expected_extension or
                values.get("data_level_extension") != expected_extension or
                int(values.get("modulus_entries", "-1")) !=
                int(values.get("baseline_entries", "-1")) + int(expected_extension)):
            raise ValueError(f"invalid one-level A4 chain marker: {run}/{party_name}")

    for (layer, ablation, softmax, relu, linear), _profile in profiles.items():
        mpc_heads = softmax + relu
        for party_name, log in logs.items():
            outputs = [
                values for values in marker_values(
                    log, "OURS_A3_A4_ATTENTION_OUTPUT ")
                if values.get("layer") == str(layer) and
                values.get("ablation", "").upper() == ablation
            ]
            if (len(outputs) != 1 or outputs[0].get("status") != "PASS" or
                    outputs[0].get("dynamic_selection") != "0" or
                    outputs[0].get("surrogate") != "0" or
                    int(outputs[0].get("mpc_heads", "-1")) != mpc_heads or
                    int(outputs[0].get("linear_heads", "-1")) != linear):
                raise ValueError(f"missing real A3/A4 output marker: {run}/{party_name}")

            conversions = [
                values for values in marker_values(
                    log, "OURS_SELECTIVE_OUTPUT_CONVERSION ")
                if values.get("layer") == str(layer)
            ]
            expected_stages = ({"a4_mpc_heads_wo", "a4_linear_heads_wo"}
                               if linear and mpc_heads else
                               {"a4_linear_heads_wo"} if linear else
                               {"a3_mpc_heads_wo"})
            real_stages = {
                values.get("stage") for values in conversions
                if values.get("status") == "PASS" and
                values.get("algorithm") == "BLB_Algorithm_1" and
                values.get("client_reencrypt") == "0" and
                values.get("ciphertexts") == "12"
            }
            if not expected_stages.issubset(real_stages):
                raise ValueError(
                    f"missing real selective output conversion: {run}/{party_name}")

        if linear:
            # The Phantom evaluator emits this marker.  The other party owns
            # MPC shares but intentionally performs no HE arithmetic.
            native = []
            for log in logs.values():
                native.extend(
                    values for values in marker_values(
                        log, "OURS_A4_NATIVE_CKKS ")
                    if values.get("layer") == str(layer)
                )
            if (len(native) != 1 or native[0].get("status") != "PASS" or
                    native[0].get("linear_heads") != str(linear) or
                    native[0].get("client_decrypt_reencrypt") != "0" or
                    native[0].get("mpc_reference") != "0" or
                    native[0].get("v_alignment_rescales") != "1"):
                raise ValueError(f"missing one-level native A4 marker: {run}")


def require_head_grouping_disabled(
        run: Path,
        profiles: dict[tuple[int, str, int, int, int], Profile]) -> None:
    """Require the allocator-only, unpacked Head execution mode.

    ``OURS_VIT_GROUP_PACKING=0`` disables the later heterogeneous Head
    packing/grouping optimization while retaining the existing A3/A4
    selective conversion and native CKKS Linear paths.  The runtime emits the
    authoritative mode marker in each party log; the launcher selection file
    is checked as an independent provenance record.
    """
    selection = key_values((run / "gpu_selection.txt").read_text(
        encoding="utf-8", errors="replace"))
    if selection.get("attention_packing") != "unpacked":
        raise ValueError(
            f"composition probe did not disable heterogeneous Head grouping: {run}")
    for party_name in ("alice.log", "bob.log"):
        log = (run / party_name).read_text(encoding="utf-8", errors="replace")
        runtime = marker_values(log, "OURS_VIT_RUNTIME boundary=padding")
        if len(runtime) != 1:
            raise ValueError(f"missing unique runtime packing marker: {run}/{party_name}")
        values = runtime[0]
        # In the current binary this metadata bit names the heterogeneous
        # grouping fast path, so it is 0 when grouping is disabled.  The
        # independent BLB-fast QK/AV markers checked below are the authoritative
        # proof that the original BLB diagonal-reuse schedule remains active.
        if (values.get("attention_packing") != "unpacked" or
                values.get("heterogeneous_head_grouping") != "0"):
            raise ValueError(
                f"composition probe used an invalid Head grouping/BLB mode: {run}/{party_name}")
        # In the ungrouped mode every selective conversion call is explicitly
        # per-head.  Keep this audit tied to the measured assignment rather
        # than accepting a global environment variable alone.
        conversions = marker_values(log, "OURS_A3_SELECTIVE_MPC_TO_CKKS")
        expected = sum(softmax + relu for _layer, _ablation, softmax, relu, _linear
                       in profiles)
        if len(conversions) != expected or any(
                values.get("grouped_call") != "0" for values in conversions):
            raise ValueError(
                f"composition probe contains grouped selective conversion calls: {run}/{party_name}")


def require_blb_fast_packing_markers(
        run: Path,
        profiles: dict[tuple[int, str, int, int, int], Profile]) -> bool:
    """Validate the original QK schedule and the selected real AV schedule.

    The allocator experiment disables only the later heterogeneous Head
    grouping.  In that mode the existing QK fast schedule remains active, but
    AV uses the genuine per-diagonal selector/rescale path and therefore emits
    ``FORMAL_AV_DIAGONAL_PACK`` instead of ``OURS_BLB_AV_PACKING``.  Both
    markers are accepted here; neither path is a secure-cost surrogate.
    """
    logs = [
        (run / party).read_text(encoding="utf-8", errors="replace")
        for party in ("alice.log", "bob.log")
    ]
    qk = []
    av = []
    ungrouped_av = []
    levels = []
    for log in logs:
        qk.extend(marker_values(log, "OURS_BLB_QK_PACKING "))
        av.extend(marker_values(log, "OURS_BLB_AV_PACKING "))
        ungrouped_av.extend(marker_values(log, "FORMAL_AV_DIAGONAL_PACK "))
        levels.extend(marker_values(log, "OURS_BLB_FAST_AV_LEVEL "))
    complete_qk = len(qk) == len(profiles) and all(
        values.get("status") == "EXECUTED" and
        values.get("mode") == "complete_diagonal" and
        values.get("mpc_surrogate") == "0" and
        values.get("client_decrypt_reencrypt") == "0"
        for values in qk)
    if not complete_qk and (len(qk) != len(profiles) or any(
            values.get("status") != "PASS" or
            values.get("mode") != "baseline_fast" or
            values.get("mpc_surrogate") != "0" or
            values.get("client_decrypt_reencrypt") != "0" or
            values.get("a0_numerical_validation") != "PENDING"
            for values in qk)):
        raise ValueError(f"assignment cost did not use BLB-fast QK: {run}")

    expected_path_count = sum(
        int(softmax + relu > 0) + int(linear > 0)
        for _layer, _ablation, softmax, relu, linear in profiles)
    packed_av_valid = len(av) == expected_path_count and all(
            values.get("status") == "PASS" and
            values.get("mode") == "baseline_fast" and
            values.get("extra_mask_multiplies") == "0" and
            values.get("extra_rescales") == "0" and
            values.get("mpc_surrogate") == "0" and
            values.get("client_decrypt_reencrypt") == "0" and
            values.get("a0_numerical_validation") == "PENDING"
            for values in av)
    ungrouped_av_valid = len(ungrouped_av) == expected_path_count and all(
        values.get("status") == "PASS" and
        int(values.get("active_pairs", "0")) > 0 and
        int(values.get("packed_diagonals", "0")) > 0 and
        int(values.get("mask_multiplies", "0")) > 0 and
        int(values.get("rescales", "0")) > 0
        for values in ungrouped_av)
    if not (packed_av_valid or ungrouped_av_valid):
        raise ValueError(
            f"assignment cost lacks a valid real AV schedule: {run}")
    if packed_av_valid and ungrouped_av:
        raise ValueError(
            f"assignment cost mixes packed and ungrouped AV schedules: {run}")
    if complete_qk:
        if not ungrouped_av_valid:
            raise ValueError(f"complete QK requires complete AV packing: {run}")
        return False
    if len(levels) != expected_path_count or any(
            values.get("status") != "PASS" or
            values.get("extra_rescales") != "0" or
            values.get("extra_multiplies") != "0"
            for values in levels):
        raise ValueError(f"assignment cost lacks BLB-fast AV level proof: {run}")
    for layer, _ablation, softmax, relu, linear in profiles:
        expected_contexts = set()
        if softmax + relu:
            expected_contexts.add("he2")
        if linear:
            expected_contexts.add("he1")
        actual_contexts = {
            values.get("context") for values in levels
            if values.get("layer") == str(layer)
        }
        if actual_contexts != expected_contexts:
            raise ValueError(
                f"BLB-fast AV contexts disagree with assignment: {run}")
    return packed_av_valid


def read_head_plan(run: Path) -> dict[int, dict[str, str]]:
    parties: list[dict[int, dict[str, str]]] = []
    for party in (1, 2):
        path = run / f"head_plan_party{party}.csv"
        rows: dict[int, list[dict[str, str]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(int(row["layer"]), []).append(row)
        serialized: dict[int, dict[str, str]] = {}
        for layer, entries in rows.items():
            if len(entries) != 12:
                raise ValueError(f"head plan layer is not 12 Heads: {path}")
            by_physical = sorted(entries, key=lambda row: int(row["physical_head"]))
            by_original = sorted(entries, key=lambda row: int(row["original_head"]))
            if ([int(row["physical_head"]) for row in by_physical] != list(range(12)) or
                    [int(row["original_head"]) for row in by_original] != list(range(12))):
                raise ValueError(f"head plan is not a permutation: {path}")
            serialized[layer] = {
                "protocols_original": ";".join(row["protocol"] for row in by_original),
                "alphas_original": ";".join(row["alpha"] for row in by_original),
                "physical_protocols": ";".join(row["protocol"] for row in by_physical),
                "original_order": ";".join(row["original_head"] for row in by_physical),
            }
        parties.append(serialized)
    if parties[0] != parties[1]:
        raise ValueError(f"party head-plan audits disagree: {run}")
    return parties[0]


def measured_rows(run: Path) -> list[dict[str, object]]:
    ablation, binary_sha256, image_id, plan_sha256, topology = \
        require_run_provenance(run)
    alice = read_party_profiles(run, "alice.log", ablation)
    bob = read_party_profiles(run, "bob.log", ablation)
    if set(alice) != set(bob):
        raise ValueError(f"party assignment profiles disagree: {run}")
    require_native_markers(run, alice)
    av_uses_fast_packing = require_blb_fast_packing_markers(run, alice)
    require_head_grouping_disabled(run, alice)
    selection = key_values((run / "gpu_selection.txt").read_text(
        encoding="utf-8", errors="replace"))
    head_plan = read_head_plan(run)
    rows: list[dict[str, object]] = []
    integer_fields = (
        "rounds", "protocol_groups", "mpc_group_calls",
        "native_linear_groups", "ckks_to_mpc", "mpc_to_ckks",
        "he_ct_pt_multiply", "he_ct_ct_multiply", "he_rotations",
        "he_rescales",
    )
    for key in sorted(alice):
        a = alice[key].values
        b = bob[key].values
        if key[0] not in head_plan:
            raise ValueError(f"assignment profile lacks head-plan audit: {run}")
        plan = head_plan[key[0]]
        protocol_counts = {
            name: plan["protocols_original"].split(";").count(name)
            for name in ("softmax_mpc", "relu_norm_mpc", "he_linear")
        }
        if ((protocol_counts["softmax_mpc"],
             protocol_counts["relu_norm_mpc"],
             protocol_counts["he_linear"]) != key[2:]):
            raise ValueError(f"profile and audited Head assignment disagree: {run}")
        row: dict[str, object] = {
            "layer": key[0], "ablation": key[1],
            "softmax_heads": key[2], "relu_heads": key[3],
            "linear_heads": key[4],
            "latency_ms": max(float(a["latency_ms"]), float(b["latency_ms"])),
            "he_instrumented_ms": max(float(a["he_instrumented_ms"]),
                                          float(b["he_instrumented_ms"])),
            "conversion_instrumented_ms": max(
                float(a["conversion_instrumented_ms"]),
                float(b["conversion_instrumented_ms"])),
            "communication_bytes": (int(a["communication_bytes"]) +
                                      int(b["communication_bytes"])),
            "measurement_scope": "complete_attention_assignment",
            "cost_unit": "direct_assignment_not_per_head",
            "network_gbps": 1.0, "rtt_ms": 0.3,
            "backend": "phantom_gpu",
            "gpu_topology": topology,
            "image_id": image_id, "binary_sha256": binary_sha256,
            "plan_sha256": plan_sha256, "surrogate": 0,
            **plan,
            "a4_data_level_extension": 1 if key[1] == "A4" else 0,
            "blb_fast_attention_packing": int(av_uses_fast_packing),
            "attention_packing": selection.get("attention_packing", ""),
            "heterogeneous_head_grouping": 0,
            "dynamic_selection": 0, "source_run": str(run.resolve()),
        }
        for field in integer_fields:
            # Calls/operators are the same logical graph on both parties;
            # rounds use the slower party. Never sum either into a fake path.
            row[field] = max(int(a[field]), int(b[field]))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for run in args.run:
        rows.extend(measured_rows(run))
    identities = [
        (row["layer"], row["plan_sha256"], row["ablation"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate directly measured assignment rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("OURS_A3_A4_COST_COLLECTION status=PASS")
    print("OURS_A3_A4_COST_UNIT complete_attention_assignment")
    print("OURS_A3_A4_COST_EXTRAPOLATION per_head=0 layer_copy=0 surrogate=0")
    print(f"OURS_A3_A4_COST_CSV={args.output}")


if __name__ == "__main__":
    main()
