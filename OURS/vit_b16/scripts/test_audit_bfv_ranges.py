#!/usr/bin/env python3
"""Self-tests for audit_bfv_ranges.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_bfv_ranges as audit
import vit_reference as reference


def entry(name: str, kind: str, values: list[int]) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "domain": "integer_raw_pre_modulus",
        "shape": [len(values)],
        "raw": values,
        "accumulator_scale": 24,
    }


class BfvRangeAuditTest(unittest.TestCase):
    def write_json(self, root: Path, payload: dict[str, object], name: str = "trace.json") -> Path:
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def base(self) -> dict[str, object]:
        return {
            "format": "BLB_VIT_FIXED_TRACE_V1",
            "plain_modulus": audit.PLAIN_MODULUS,
            "expected_accumulators": ["patch_projection", "block_0.qk"],
            "accumulators": [
                entry("patch_projection", "linear", [-7, 0, 9]),
                entry("block_0.qk", "qk", [-11, 13]),
            ],
        }

    def test_inline_json_passes_at_centered_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.base()
            payload["accumulators"] = [
                entry("patch_projection", "linear",
                      [-audit.CENTERED_HALF, audit.CENTERED_HALF]),
                entry("block_0.qk", "qk", [0]),
            ]
            report, rows = audit.audit(self.write_json(root, payload))
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(all(row.status == "PASS" for row in rows))

    def test_wrap_is_fail_not_numerical_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.base()
            payload["accumulators"][1] = entry(
                "block_0.qk", "qk", [audit.CENTERED_HALF + 1])
            report, rows = audit.audit(self.write_json(root, payload))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["first_failed_accumulator"], "block_0.qk")
            failed = next(row for row in rows if row.name == "block_0.qk")
            self.assertTrue(failed.modular_wrap)
            self.assertIn("modular_wrap", failed.reason)
            self.assertNotIn("error", failed.reason)

    def test_npz_with_embedded_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "format": "BLB_VIT_FIXED_TRACE_V1",
                "plain_modulus": audit.PLAIN_MODULUS,
                "expected_accumulators": ["patch_projection", "block_0.qk"],
                "accumulators": [
                    {"name": "patch_projection", "kind": "linear",
                     "domain": "integer_raw_pre_modulus", "shape": [3],
                     "npz_key": "raw/patch_projection", "accumulator_scale": 24},
                    {"name": "block_0.qk", "kind": "qk",
                     "domain": "integer_raw_pre_modulus", "shape": [2, 2],
                     "npz_key": "raw/block_0.qk", "accumulator_scale": 24},
                ],
            }
            path = root / "trace.npz"
            np.savez(path,
                     __metadata__=json.dumps(metadata),
                     **{"raw/patch_projection": np.asarray([-1, 2, 3], dtype=np.int64),
                        "raw/block_0.qk": np.asarray([[-4, 5], [6, -7]], dtype=np.int64)})
            report, rows = audit.audit(path)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(rows), 2)

    def test_unsigned_modular_npz_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "format": "BLB_VIT_FIXED_TRACE_V1",
                "plain_modulus": audit.PLAIN_MODULUS,
                "expected_accumulators": ["wrapped"],
                "accumulators": [{
                    "name": "wrapped", "kind": "linear",
                    "domain": "integer_raw_pre_modulus", "shape": [1],
                    "npz_key": "raw/wrapped",
                }],
            }
            path = root / "trace.npz"
            np.savez(path, __metadata__=json.dumps(metadata),
                     **{"raw/wrapped": np.asarray([audit.PLAIN_MODULUS - 1], dtype=np.uint64)})
            with self.assertRaisesRegex(audit.AuditInputError, "signed-integer"):
                audit.audit(path)

    def test_current_reference_report_fails_closed_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.write_json(root, {
                "format": "BOLT_VIT_REFERENCE_REPORT_V1",
                "operators": [{"name": "patch_projection", "fixed": {"scale": 12}}],
            })
            csv_path = root / "audit.csv"
            json_path = root / "audit.json"
            status = audit.main([
                "--trace", str(trace), "--csv", str(csv_path),
                "--json", str(json_path),
            ])
            self.assertEqual(status, 1)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("no pre-modulus raw accumulators", report["schema_error"])
            self.assertIn("__trace__", csv_path.read_text(encoding="utf-8"))

    def test_missing_expected_accumulator_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.base()
            payload["accumulators"] = payload["accumulators"][:1]
            with self.assertRaisesRegex(audit.AuditInputError, "missing"):
                audit.audit(self.write_json(root, payload))

    def test_unexpected_accumulator_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.base()
            payload["accumulators"].append(entry("uncovered", "linear", [0]))
            with self.assertRaisesRegex(audit.AuditInputError, "unexpected"):
                audit.audit(self.write_json(root, payload))

    def test_vit_reference_trace_is_complete_exact_and_auditable(self) -> None:
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            trace_dir = root / "trace"
            subprocess.run([
                sys.executable, str(scripts / "generate_vit_fixture.py"),
                "--out-dir", str(fixture), "--seed", "7", "--layers", "1",
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, str(scripts / "vit_reference.py"),
                "--model", str(fixture / "alice_model.txt"),
                "--input", str(fixture / "bob_image.txt"),
                "--output", str(root / "reference.json"),
                "--trace-dir", str(trace_dir),
            ], check=True, capture_output=True, text=True)

            trace = trace_dir / reference.BFV_TRACE_JSON
            metadata = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(metadata["format"], reference.BFV_TRACE_FORMAT)
            self.assertEqual(metadata["coverage"], {
                "expected": 13, "observed": 13, "missing": [],
                "unexpected": [], "complete": True,
            })
            expected_names = [item["name"]
                              for item in metadata["expected_accumulators"]]
            self.assertEqual(expected_names, [
                "patch_projection", "block_0.ln1_affine",
                "block_0.q_projection",
                "block_0.k_projection", "block_0.v_projection",
                "block_0.qk", "block_0.softmax_v",
                "block_0.output_projection", "block_0.ln2_affine",
                "block_0.fc1", "block_0.fc2", "final_ln_affine",
                "classification_head",
            ])
            report, rows = audit.audit(trace)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(rows), 13)

            model = reference.load_model(fixture / "alice_model.txt")
            pixels = reference.load_input(fixture / "bob_image.txt")
            patches = reference.quantized(
                reference.patchify(pixels, int(model["patch_size"])),
                reference.ACT_SCALE)
            direct = reference.matmul_integer(
                reference.round_int(patches, reference.ACT_SCALE),
                reference.round_int(model["patch_weight"],
                                    reference.WEIGHT_SCALE).T)
            direct += reference.round_int(
                model["patch_bias"],
                reference.ACT_SCALE + reference.WEIGHT_SCALE)
            direct = reference.signed_raw_accumulator(
                direct, "patch_projection")
            patch_entry = metadata["accumulators"][0]
            with np.load(trace_dir / reference.BFV_TRACE_ARRAYS,
                         allow_pickle=False) as archive:
                stored = archive[patch_entry["npz_key"]]
                np.testing.assert_array_equal(stored, direct)
                self.assertEqual(stored.dtype, np.dtype(np.int64))

            tampered = dict(metadata)
            tampered["accumulators"] = [dict(item)
                                        for item in metadata["accumulators"]]
            tampered["accumulators"][0]["raw_sha256"] = "0" * 64
            tampered_path = trace_dir / "tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditInputError, "SHA-256 mismatch"):
                audit.audit(tampered_path)


if __name__ == "__main__":
    unittest.main()
