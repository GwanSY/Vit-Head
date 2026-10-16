"""Finalize after the serial benchmark driver; no new experiments are launched."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from runtime import ROOT, write


def finish(out, wait):
    with (out / ".finish.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            meta=json.loads((out / "run.json").read_text())
            if meta["status"] in ("experiments_complete","complete"): break
            if meta["status"] in ("failed","interrupted"):
                raise RuntimeError("Experiment driver stopped: "+meta.get("error",meta["status"]))
            if not wait: raise RuntimeError("Experiments still running")
            time.sleep(10)
        for script in ("optimize_analysis.py","optimize_report.py","optimize_package.py"):
            print("Running",script,flush=True)
            with (out / (script.replace(".py","")+".log")).open("w") as log:
                subprocess.run([sys.executable,str(ROOT / script),"--out",str(out)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        print("All evidence, report, vector figures and raw archive verified",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args();finish(args.out.resolve(),args.wait)
