"""ViT-only baselines, accuracy, native builds, and real secure cost experiments."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "worker", "node", "build", "accuracy", "costs", "allocate", "full", "select", "study"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rank", type=int, choices=[0, 1])
    parser.add_argument("--backend", choices=["cpu", "gpu"])
    parser.add_argument("--native", type=Path)
    parser.add_argument("--plan", default="R12")
    parser.add_argument("--layout", choices=["joint", "split"], default="joint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("out") / "07")
    parser.add_argument("--stage", choices=["all", "cost", "heads", "acc", "holdout", "full", "report"], default="all")
    args = parser.parse_args()
    if args.command != "build" and args.command != "baseline" and args.out is None:
        parser.error("--out is required")
    if args.command == "allocate" and args.costs is None:
        parser.error("allocate requires --costs")
    backend = args.backend or "cpu"
    native = args.native or Path(__file__).with_name("build") / f"{backend}.so"
    from runtime import baseline, node, worker
    if args.command == "build":
        from build import build
        build(args.backend or "cpu")
    elif args.command == "accuracy":
        from accuracy import run
        run(args.out.resolve(), args.backend or "cpu")
    elif args.command == "costs":
        from experiment import costs
        costs(args.out.resolve(), backend, native.resolve(), args.data.resolve())
    elif args.command == "allocate":
        from experiment import allocate
        allocate(args.out.resolve(), args.costs.resolve(), args.budget, backend, args.data.resolve())
    elif args.command == "full":
        from experiment import full
        full(args.out.resolve(), args.plan, backend, native.resolve(), args.repeats, args.layout, args.data.resolve())
    elif args.command == "select":
        from experiment import select
        select(args.out.resolve(), args.data.resolve())
    elif args.command == "study":
        from study import run
        if backend!="cpu":parser.error("This study uses CPU only")
        run(args.out.resolve(),args.stage,args.data.resolve(),native.resolve())
    elif args.command == "baseline":
        try:
            baseline(args.config.resolve(), args.out, args.backend)
        except KeyboardInterrupt:
            parser.exit(130, "Stopped; details saved in run.json.\n")
        except Exception as exc:
            parser.exit(1, f"Failed: {type(exc).__name__}: {str(exc)[:240]}\n")
    elif args.command == "worker":
        worker(args.out)
    else:
        node(args.out, args.rank)


if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped; partial evidence retained in log/.")
    except Exception as exc:
        if len(sys.argv) > 1 and sys.argv[1] in ("worker", "node"):
            raise
        raise SystemExit(f"Failed: {type(exc).__name__}: {str(exc)[:240]}")
