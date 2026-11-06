"""Index every run under a checkpoints folder: its config and its best model path.

Usage (on the machine that holds the checkpoints, from the repo root):

    python envs/iclr_rebutal/index_runs.py ./checkpoints
    python envs/iclr_rebutal/index_runs.py ./checkpoints --phase PRETRAIN      # filter
    python envs/iclr_rebutal/index_runs.py ./checkpoints -o runs_index.csv

Writes <out>.csv (one row per run: identity columns, paths, then EVERY key of the
run's config as its own column) and <out>.json (the same plus the config as a
nested object). A run is any folder holding a
custom_config.json, which the server writes at start-up.

If envs/iclr_rebutal/manifest.csv exists, runs created from these rebuttal envs
are matched by CONFIG_ID and get their row (A0, A3, ...), level, scenario, etc.
"""
import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_manifest() -> dict:
    f = HERE / "manifest.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8", newline="") as fh:
        return {r["config_id"]: r for r in csv.DictReader(fh)}


def first_value(d):
    return list(d.values())[0] if isinstance(d, dict) and d else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="./checkpoints")
    ap.add_argument("-o", "--out", default="runs_index", help="output name without extension")
    ap.add_argument("--phase", help="keep only this STARTING_PHASE (PRETRAIN, MAX, LEARN, EXACT, POST, ...)")
    args = ap.parse_args()

    manifest = load_manifest()
    runs, full = [], []
    for cfg_file in sorted(Path(args.root).rglob("custom_config.json")):
        run_dir = cfg_file.parent
        try:
            j = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception as e:  # unreadable config: still list the run
            print("skip (bad json):", cfg_file, e)
            continue
        phase = str(j.get("STARTING_PHASE", "")).strip('"')
        if args.phase and phase.upper() != args.phase.upper():
            continue
        ck = run_dir / "models_chkpts"
        best, latest = ck / "model_best.pth", ck / "model_latest.pth"
        m = manifest.get(str(j.get("CONFIG_ID")), {})
        level = "client_level" if j.get("Client_ID_TO_EXIT") else "data_level"
        frac = first_value(j.get("FORGET_CLASS"))
        row = {
            "row": m.get("row", ""),
            "phase": phase,
            "level": level,
            "scenario": str(j.get("UNLEARNING_CASE", "")).strip('"').lower(),
            "clients": j.get("NUM_SUPERNODES", ""),
            "percent": "" if level == "client_level" or frac == "" else round(float(frac) * 100),
            "divergence": j.get("LOSSDIV", ""),
            "config_id": j.get("CONFIG_ID", ""),
            "config_number": j.get("CONFIG_NUMBER", ""),
            "seed": j.get("SEED", ""),
            "rounds": j.get("NUM_ROUNDS", ""),
            "post_phase": j.get("POST_PHASE", ""),
            "alpha": j.get("ALPHA", ""),
            "gamma": j.get("GAMMA", ""),
            "resumed_from": j.get("RESUME", ""),
            "best_model": best.as_posix() if best.exists() else "",
            "latest_model": latest.as_posix() if latest.exists() else "",
            "logs_csv": (run_dir / "logs.csv").as_posix() if (run_dir / "logs.csv").exists() else "",
            "run_dir": run_dir.as_posix(),
        }
        runs.append(row)
        full.append({**row, "config": j})

    if not runs:
        print("no runs found under", args.root)
        return
    # CSV: identity + paths first, then every config key (lists/dicts as JSON text).
    head = list(runs[0].keys())
    cfg_keys = sorted({k for r in full for k in r["config"]} - set(head))
    with open(f"{args.out}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(head + cfg_keys)
        for r in full:
            cells = [json.dumps(v) if isinstance(v, (list, dict)) else v
                     for v in (r["config"].get(k, "") for k in cfg_keys)]
            w.writerow([r[k] for k in head] + cells)
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    with_best = sum(1 for r in runs if r["best_model"])
    print(f"{len(runs)} runs, {with_best} with model_best.pth -> {args.out}.csv, {args.out}.json")


if __name__ == "__main__":
    main()
