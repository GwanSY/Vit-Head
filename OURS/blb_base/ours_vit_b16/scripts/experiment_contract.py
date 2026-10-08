"""Scientific definitions shared by offline exports and the secure launcher.

This describes required semantics; it is not a certificate of runtime accuracy
or cryptographic security. Those still require the native audit and validation.
"""
import hashlib
import json


CONTRACT = {
    "schema": "OURS_EXPERIMENT_CONTRACT_V1",
    "adaptation": "offline_model_accuracy_budget_public_deployment",
    "online_execution": "EdgeCloudRuntime_fixed_public_plan",
    "calibration_data": "model_party_authorized_training_or_calibration_only",
    "functions": {
        "Z": "Q @ K.T / sqrt(head_dim); valid keys only",
        "S": "softmax(Z) @ V",
        "R": "max(Z,0) @ V / sum_key(max(Z,0)); zero denominator => zero output; no epsilon",
        "L": "alpha[layer,original_head] * (Z @ V); signed; no row normalization",
        "alpha": "calibration least squares sum(<ZV,softmax(Z)V>)/sum(||ZV||^2); nonpositive denominator rejected; frozen online",
    },
    "security": {
        "ckks_security_target": "tc128",
        "nonlinear_ring_bits": 37,
        "nonlinear_fractional_bits": 12,
        "same_security_and_numerical_acceptance_all_versions": True,
        "shape_specific_parameters": "allowed under same security target and numerical acceptance; record actual parameters",
        "paired_packing": "freeze function, alpha, precision and approximation quality",
    },
    "cost": {
        "objective": "measured_complete_attention_assignment_max_party_wall_ms",
        "covered_in_wall": ["HE", "MPC", "conversions", "online_repacking", "communication_waits", "edge_execution"],
        "audit": ["HE_operations", "MPC_calls", "conversion_ciphertexts", "rounds", "party_wall_ms", "offline_head_reorder_ms"],
        "separate_sync_timer": "unavailable; included in wall within measurement scope",
        "offline_reorder": "report separately; excluded from online objective",
        "layer_sum": "prediction for attention only; not measured full-network latency",
    },
    "search": {
        "name": "准确率约束下的高效头配置算法",
        "global_optimality_claimed": False,
        "accuracy": "whole-model calibration validation; head proxies are not additive accuracy guarantees",
        "selection_scope": "retained calibration-verified candidates under predicted cost",
        "small_oracle": "optimal only within explicitly enumerated space and declared objective",
    },
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode()).hexdigest()

ONLINE_INVARIANTS = {
    "fixed_public_assignment": True,
    "dynamic_selection": False,
    "private_dynamic_branch": False,
    "mpc_reference": 0,
    "client_decrypt_reencrypt": 0,
    "plaintext_search_at_inference": False,
}


def validate_assignment_contract(document):
    invariants = document.get("online_invariants", {})
    for key, expected in ONLINE_INVARIANTS.items():
        value = invariants.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"assignment violates offline-fixed contract: {key}")
    if (document.get("experiment_contract_sha256") != CONTRACT_SHA256 or
            document.get("experiment_contract") != CONTRACT):
        raise ValueError("assignment experiment definitions missing or incompatible; re-export offline")
