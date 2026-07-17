#!/usr/bin/env python
"""Average GPU utilization over a wandb training run.

wandb logs per-GPU utilization as the system metric ``system.gpu.<N>.gpu`` (a
percentage, 0-100). This script pulls that system-metrics history for one or
more runs and reports the time-averaged utilization, per GPU and overall.

Usage
-----
    # bare run id (uses the default entity/project below)
    python gpu_util.py dinogcbc_4899612

    # several runs at once
    python gpu_util.py dinogcbc_4899612 dinogcbc_4893582

    # fully-qualified path overrides the defaults
    python gpu_util.py someentity/someproject/somerunid

    # different default project
    python gpu_util.py --project worldbench-world worldmodel_12345

Notes
-----
* The average is over logged samples, which wandb records at a fixed cadence
  (~every 15 s), so it is effectively a time-average of the run's lifetime.
* For a run that is still ``running`` the number is the average *so far*.
* ``system.gpu.0.gpu`` is core utilization. ``system.gpu.0.memory`` (also
  reported here) is the memory-controller busy percentage, not GB used.
"""

import argparse
import re

import wandb


DEFAULT_ENTITY = "ahmad-baki-karlsruhe-institute-of-technology"
DEFAULT_PROJECT = "worldbench-bc"

# matches e.g. "system.gpu.0.gpu" / "system.gpu.3.gpu" -> the utilization %
UTIL_RE = re.compile(r"^system\.gpu\.(\d+)\.gpu$")


def resolve_path(run_arg, entity, project):
    """Turn a bare id or an ``entity/project/id`` string into a full path."""
    parts = run_arg.split("/")
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) == 1:
        return f"{entity}/{project}/{run_arg}"
    raise ValueError(
        f"'{run_arg}' is neither a bare run id nor an 'entity/project/id' path"
    )


def gpu_util_stats(run, samples):
    """Return per-GPU utilization stats for a run.

    Returns a dict with keys: ``per_gpu`` (idx -> stats dict), ``overall``
    (stats dict averaged across GPUs), ``n_samples``.
    """
    hist = run.history(stream="system", samples=samples)

    util_cols = sorted(
        (c for c in hist.columns if UTIL_RE.match(c)),
        key=lambda c: int(UTIL_RE.match(c).group(1)),
    )
    if not util_cols:
        return None

    per_gpu = {}
    for col in util_cols:
        idx = int(UTIL_RE.match(col).group(1))
        series = hist[col].dropna()
        mem_col = f"system.gpu.{idx}.memory"
        mem_series = hist[mem_col].dropna() if mem_col in hist.columns else None
        per_gpu[idx] = {
            "mean": series.mean(),
            "min": series.min(),
            "max": series.max(),
            "n": int(series.count()),
            "mem_mean": mem_series.mean() if mem_series is not None else float("nan"),
        }

    # overall = mean of the per-GPU means (each GPU weighted equally)
    means = [s["mean"] for s in per_gpu.values()]
    mem_means = [s["mem_mean"] for s in per_gpu.values()]
    overall = {
        "mean": sum(means) / len(means),
        "min": min(s["min"] for s in per_gpu.values()),
        "max": max(s["max"] for s in per_gpu.values()),
        "mem_mean": sum(mem_means) / len(mem_means),
    }
    return {"per_gpu": per_gpu, "overall": overall, "n_samples": len(hist)}


def report(run, stats):
    runtime = run.summary.get("_runtime")
    runtime_str = f"{runtime / 60:.1f} min" if runtime else "n/a"
    print(f"\n{run.name}  [{run.state}]  runtime={runtime_str}")
    print(f"  {run.entity}/{run.project}/{run.id}")

    if stats is None:
        print("  no GPU system metrics found for this run")
        return

    ng = len(stats["per_gpu"])
    print(f"  samples={stats['n_samples']}  gpus={ng}")
    for idx, s in stats["per_gpu"].items():
        print(
            f"    gpu{idx}: util avg {s['mean']:5.1f}%  "
            f"(min {s['min']:.0f} / max {s['max']:.0f})  "
            f"mem-ctrl avg {s['mem_mean']:5.1f}%  n={s['n']}"
        )
    o = stats["overall"]
    print(
        f"  >> avg GPU utilization: {o['mean']:.1f}%"
        + (f"  (across {ng} GPUs)" if ng > 1 else "")
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="run id(s) or entity/project/id path(s)")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="max system-history rows to pull (default: 100000 ~= all)",
    )
    args = ap.parse_args()

    api = wandb.Api()
    for run_arg in args.runs:
        path = resolve_path(run_arg, args.entity, args.project)
        try:
            run = api.run(path)
        except Exception as e:  # noqa: BLE001 - report and continue to next run
            print(f"\n{run_arg}: could not load ({e})")
            continue
        stats = gpu_util_stats(run, args.samples)
        report(run, stats)


if __name__ == "__main__":
    main()
