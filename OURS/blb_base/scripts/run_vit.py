#!/usr/bin/env python3
"""ViT entry point: experiment.json, full inference by default."""
import runpy
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parents[1] / "ours_vit_b16/scripts"
sys.path.insert(0, str(scripts))
if not any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv[1:]):
    sys.argv.extend(["--stage", "secure"])
runpy.run_path(str(scripts / "run_research2.py"), run_name="__main__")
