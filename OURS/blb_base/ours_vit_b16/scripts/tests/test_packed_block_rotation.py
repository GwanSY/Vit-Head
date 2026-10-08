import subprocess
import tempfile
import unittest
from pathlib import Path


class PackedBlockRotationTest(unittest.TestCase):
    def test_signed_heads_never_mix_for_all_block_positions(self):
        source = Path(__file__).resolve().parents[2] / 'runtime/packed_block_rotation_test.cpp'
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'rotation-test')
            subprocess.run(['g++', '-O2', '-std=c++17', '-fsanitize=undefined',
                            '-fno-sanitize-recover=undefined', str(source), '-o', binary], check=True)
            result = subprocess.run([binary], text=True, capture_output=True, check=True)
            self.assertIn('status=PASS', result.stdout)
            self.assertFalse(result.stderr)
