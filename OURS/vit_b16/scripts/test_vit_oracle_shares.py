#!/usr/bin/env python3
"""Offline unit tests for role-separated ViT validation oracle shares."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import prepare_vit_oracle_shares as oracle
from vit_reference import Graph, load_input, load_model


class VitOracleShareTest(unittest.TestCase):
    def test_fixture_shares_reconstruct_and_are_randomized(self) -> None:
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            shares = root / "shares"
            subprocess.run([
                sys.executable, str(scripts / "generate_vit_fixture.py"),
                "--out-dir", str(fixture), "--seed", "19", "--layers", "1",
                "--protocol", "BOLT",
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, str(scripts / "prepare_vit_oracle_shares.py"),
                "--model", str(fixture / "alice_model.txt"),
                "--input", str(fixture / "bob_image.txt"),
                "--output", str(shares), "--block-limit", "1",
            ], check=True, capture_output=True, text=True)

            manifest = json.loads((shares / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], "blb.vit.validation-oracle.v2")
            self.assertFalse(
                manifest["intermediate_reconstruction_by_online_parties"])
            self.assertEqual(len(manifest["stages"]), 12)
            model = load_model(fixture / "alice_model.txt")
            pixels = load_input(fixture / "bob_image.txt")
            _, clear = Graph(model, fixed=False, block_limit=1).run(pixels)
            _, fixed = Graph(model, fixed=True, block_limit=1).run(pixels)

            for stage in manifest["stages"]:
                name = stage["name"]
                for kind, records in (("fixed", fixed), ("clear", clear)):
                    role_values = []
                    for role in ("alice", "bob"):
                        metadata = stage["shares"][kind][role]
                        path = shares / metadata["file"]
                        self.assertEqual(
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                            metadata["sha256"])
                        data = path.read_bytes()
                        self.assertTrue(data.startswith(oracle.MAGIC))
                        offset = len(oracle.MAGIC)
                        elements, ell, scale = struct.unpack_from("<Qii", data, offset)
                        self.assertEqual(elements, stage["elements"])
                        self.assertEqual((ell, scale),
                                         (oracle.ORACLE_ELL, oracle.ORACLE_SCALE))
                        role_values.append(np.frombuffer(
                            data, dtype=np.uint64,
                            offset=offset + struct.calcsize("<Qii")).copy())
                    self.assertTrue(np.any(role_values[0]))
                    reconstructed = (role_values[0] + role_values[1]) & np.uint64(
                        oracle.ORACLE_MASK)
                    value = oracle.valid_value(
                        name, records[name]["value"], int(model["tokens"]))
                    expected = oracle.encoded_reference(
                        kind, value, int(stage["source_scale"]))
                    np.testing.assert_array_equal(reconstructed, expected)


if __name__ == "__main__":
    unittest.main()
