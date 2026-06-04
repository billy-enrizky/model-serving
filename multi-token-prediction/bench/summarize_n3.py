#!/usr/bin/env python3
"""Aggregate n=3 re-bench cells into mean + spread per (engine, gpu, regime, mode).

Each cell is run 3 times as independent cold invocations labelled
  <ts>_<base>_r1 / _r2 / _r3
where <base> is e.g. vllm_mtp_h100_structured_c1 or
transformers_mtp_const_h100_structured_c1 or baseline_n0_h100_structured_c1.

Throughput is read from aggregate.warm_only.system_throughput_tokens_per_sec
(idx=0 cold cohort dropped, the same number the README headline tables cite).

Acceptance source differs per engine:
  - transformers: result.json aggregate.speculative_decoding.overall_acceptance_rate
  - vLLM:         vllm_metrics.prom  accepted_total / draft_tokens_total
    (the harness speculative_decoding block is 0 for vLLM; vLLM reports via /metrics)

For mtp/baseline pairs that share (engine, gpu, regime), the mtp/baseline ratio
is computed PER RUN-INDEX (r1/r1, r2/r2, r3/r3) and reported as mean +/- stddev
of those paired ratios, so near-breakeven cells show whether the ratio's spread
crosses 1.0.

Usage:
  uv run python bench/summarize_n3.py                 # all _r{1,2,3} cells
  uv run python bench/summarize_n3.py --runs-dir metrics/runs
  uv run python bench/summarize_n3.py --json out.json # also dump machine-readable
"""
import argparse
import glob
import json
import logging
import os
import re
import statistics
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# <ts>_<base>_r<N>  e.g. 20260603T101500_vllm_mtp_h100_structured_c1_r2
DIR_RE = re.compile(r"^(?P<ts>\d{8}T\d{6})_(?P<base>.+)_r(?P<run>\d+)$")


def parse_cell(base: str):
    """Split a base label into (engine, mode, gpu, regime).

    Bases:
      vllm_mtp_<gpu>[_<regime>]_c1
      vllm_baseline_<gpu>[_<regime>]_c1
      transformers_mtp_const_<gpu>[_<regime>]_c1
      mtp_n4_<gpu>[_<regime>]_c1            (transformers mtp, older A/B label)
      baseline_n0_<gpu>[_<regime>]_c1       (transformers baseline)
    regime defaults to 'generic' when no _<regime> infix is present.
    """
    regimes = ("generic", "code", "structured")
    # Strip trailing concurrency tag _c<N>.
    b = re.sub(r"_c\d+$", "", base)
    if b.startswith("vllm_"):
        engine = "vllm"
        b = b[len("vllm_"):]
        mode = "mtp" if b.startswith("mtp") else "baseline"
        b = b[len("mtp"):] if b.startswith("mtp") else b[len("baseline"):]
    elif b.startswith("transformers_mtp_const_"):
        engine, mode = "transformers", "mtp"
        b = b[len("transformers_mtp_const_"):]
    elif b.startswith("mtp_n4_"):
        engine, mode = "transformers", "mtp"
        b = b[len("mtp_n4_"):]
    elif b.startswith("baseline_n0_"):
        engine, mode = "transformers", "baseline"
        b = b[len("baseline_n0_"):]
    else:
        return None
    b = b.strip("_")
    regime = "generic"
    for rg in regimes:
        if b == rg or b.endswith("_" + rg):
            regime = rg
            b = b[: -len(rg)].strip("_") if b != rg else ""
            break
    gpu = b.strip("_")
    return engine, mode, gpu, regime


def vllm_acceptance(prom_path: str):
    """accepted_tokens_total / draft_tokens_total from a vLLM /metrics scrape."""
    if not os.path.exists(prom_path):
        return None
    accepted = drafted = None
    with open(prom_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "spec_decode_num_accepted_tokens_total" in line:
                accepted = float(line.rsplit(None, 1)[-1])
            elif "spec_decode_num_draft_tokens_total" in line:
                drafted = float(line.rsplit(None, 1)[-1])
    if accepted is None or not drafted:
        return None
    return accepted / drafted


def load_run(path: str, engine: str):
    """Return (warm_tps, acceptance) for one run dir, or None if unusable."""
    rj = os.path.join(path, "result.json")
    if not os.path.exists(rj):
        return None
    agg = json.load(open(rj)).get("aggregate", {})
    warm = agg.get("warm_only", {})
    tps = warm.get("system_throughput_tokens_per_sec")
    if tps is None:
        return None
    if engine == "vllm":
        accept = vllm_acceptance(os.path.join(path, "vllm_metrics.prom"))
    else:
        accept = agg.get("speculative_decoding", {}).get("overall_acceptance_rate")
    return {"tps": tps, "accept": accept}


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "sd": sd, "median": statistics.median(xs), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="metrics/runs")
    ap.add_argument("--json", default=None, help="optional path to dump JSON")
    args = ap.parse_args()

    # cell key -> {run_index -> {tps, accept}}
    cells = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*_r[0-9]*"))):
        if not os.path.isdir(d):
            continue
        m = DIR_RE.match(os.path.basename(d))
        if not m:
            continue
        cell = parse_cell(m.group("base"))
        if cell is None:
            continue
        run_idx = int(m.group("run"))
        rec = load_run(d, cell[0])
        if rec is None:
            logger.warning("skip (no warm tps): %s", d)
            continue
        cells[cell][run_idx] = rec

    # Aggregate per cell.
    records = {}
    for cell, runs in sorted(cells.items()):
        engine, mode, gpu, regime = cell
        tps = mean_sd([r["tps"] for r in runs.values()])
        acc = mean_sd([r["accept"] for r in runs.values()])
        records["|".join(cell)] = {
            "engine": engine, "mode": mode, "gpu": gpu, "regime": regime,
            "n": len(runs), "runs": sorted(runs.keys()),
            "warm_tps": tps, "acceptance": acc,
        }

    # Pair mtp/baseline per (engine, gpu, regime); ratio per matched run index.
    ratios = {}
    for (engine, mode, gpu, regime), runs in cells.items():
        if mode != "mtp":
            continue
        base_runs = cells.get((engine, "baseline", gpu, regime))
        if not base_runs:
            continue
        paired = []
        for ri in sorted(set(runs) & set(base_runs)):
            b = base_runs[ri]["tps"]
            if b:
                paired.append(runs[ri]["tps"] / b)
        if paired:
            ratios[(engine, gpu, regime)] = {"per_run": paired, **(mean_sd(paired) or {})}

    # Report.
    logger.info("=== n=3 per-cell warm throughput (tok/s) + acceptance ===")
    for key, r in records.items():
        t = r["warm_tps"]
        a = r["acceptance"]
        acc_s = f"{a['mean']*100:.1f}+/-{a['sd']*100:.1f}%" if a else "n/a"
        logger.info(
            "  %-12s %-9s %-10s %-10s n=%d  tps=%.2f +/- %.2f (med %.2f)  accept=%s",
            r["engine"], r["mode"], r["gpu"], r["regime"], r["n"],
            t["mean"], t["sd"], t["median"], acc_s,
        )

    logger.info("\n=== n=3 mtp/baseline ratio (paired per run, mean +/- sd) ===")
    for (engine, gpu, regime), rr in sorted(ratios.items()):
        per = ", ".join(f"{x:.2f}" for x in rr["per_run"])
        logger.info(
            "  %-12s %-10s %-10s  ratio=%.3fx +/- %.3f (med %.3fx)  runs=[%s]",
            engine, gpu, regime, rr["mean"], rr["sd"], rr["median"], per,
        )

    if args.json:
        out = {
            "cells": records,
            "ratios": {"|".join(k): v for k, v in ratios.items()},
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("\nwrote %s", args.json)


if __name__ == "__main__":
    main()
