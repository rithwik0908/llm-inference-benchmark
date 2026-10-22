"""Record hardware/software without loading model weights."""

import argparse

from inference_bench.common import identity, provenance, require_cuda, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-vram-gib", type=float, default=0)
    parser.add_argument("--output", default="results/preflight.json")
    args = parser.parse_args()
    report = {"schema_version": 1, "identity": identity(), "environment": provenance()}
    try:
        report["gpu"] = require_cuda(args.min_vram_gib)
        report["cuda_ready"] = True
    except (ImportError, RuntimeError) as exc:
        report.update(cuda_ready=False, error=str(exc))
    write_json(args.output, report)
    print(f"CUDA ready: {report['cuda_ready']}; report: {args.output}")
    if not report["cuda_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
