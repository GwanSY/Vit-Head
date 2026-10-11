"""One entry point; the all-Softmax baseline must pass before research changes."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "worker", "node"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rank", type=int, choices=[0, 1])
    parser.add_argument("--backend", choices=["cpu", "gpu"])
    args = parser.parse_args()
    from runtime import baseline, node, worker
    if args.command == "baseline":
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
    main()
