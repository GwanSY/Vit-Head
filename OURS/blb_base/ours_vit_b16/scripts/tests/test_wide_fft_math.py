import random
import subprocess
import tempfile
import unittest
from pathlib import Path


class WideFftMathTest(unittest.TestCase):
    def test_matches_unbounded_integer_oracle(self):
        source = Path(__file__).resolve().parents[2] / 'runtime/wide_fft_math_test.cpp'
        rng = random.Random(0x349849)
        inputs, expected = [], []
        for width in (2, 12, 37, 40, 49, 62, 77, 83, 93, 100, 110, 126):
            for scale in (16, 32, 40, 62):
                mask, unit = (1 << width) - 1, 1 << scale
                for index in range(1000):
                    a, b = rng.getrandbits(width), rng.getrandbits(width)
                    c, d = rng.randint(-unit, unit), rng.randint(-unit, unit)
                    if index < 16:
                        a = mask if index & 1 else 1 << (width - 1)
                        b = 0 if index & 2 else mask
                        c = -unit if index & 4 else unit
                        d = 0 if index & 8 else unit
                    inputs.append(f'{width} {scale} {a >> 64} {a & ((1 << 64)-1)} '
                                  f'{b >> 64} {b & ((1 << 64)-1)} {c} {d}\n')
                    a = a - (1 << width) if a >> (width - 1) else a
                    b = b - (1 << width) if b >> (width - 1) else b
                    expected.append((((a*c - b*d) >> scale) & mask,
                                     ((a*d + b*c) >> scale) & mask))
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'fft-test')
            subprocess.run(['g++', '-O3', '-std=c++17', '-fsanitize=undefined',
                            '-fno-sanitize-recover=undefined', str(source), '-o', binary], check=True)
            result = subprocess.run([binary], input=''.join(inputs), text=True,
                                    capture_output=True, check=True)
        actual = []
        for line in result.stdout.splitlines():
            ah, al, bh, bl = map(int, line.split())
            actual.append(((ah << 64) | al, (bh << 64) | bl))
        self.assertEqual(actual, expected)
        self.assertFalse(result.stderr)


if __name__ == '__main__':
    unittest.main()
