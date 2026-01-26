import os
import re
import argparse
from pathlib import Path
from typing import Dict
from dotenv import dotenv_values
from ast import literal_eval

INT_KEYS = [
    "RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH",
    "NUM_CLASSES", "LOCAL_EPOCHS", "MIN_EPOCHS", "MAX_EPOCHS", "LAST_MAX_STEPS"
]


def load_config(env_dir: Path) -> Dict:
    """Load config from .env and .env.training files."""
    env_path = env_dir / ".env"
    training_path = env_dir / ".env.training"

    if not env_path.exists() and not training_path.exists():
        return {}

    config = {}
    if env_path.exists():
        config.update(dotenv_values(env_path))
    if training_path.exists():
        config.update(dotenv_values(training_path))

    # Process values
    if "FORGET_CLASS" in config:
        config["FORGET_CLASS"] = literal_eval(config["FORGET_CLASS"])
    if "MAP_CONFUSE" in config:
        config["MAP_CONFUSE"] = literal_eval(config["MAP_CONFUSE"])

    for key in INT_KEYS:
        if key in config:
            config[key] = int(config[key])

    for key in ["CLIENT_ID_TO_FORGET", "LR_ROUND", "Client_ID_TO_EXIT"]:
        if key in config and config[key]:
            config[key] = [int(i) for i in str(config[key]).split(",")]
        elif key in config:
            config[key] = []

    # Canonicalize
    if "Client_ID_TO_EXIT" not in config and "CLIENT_ID_TO_EXIT" in config:
        config["Client_ID_TO_EXIT"] = config["CLIENT_ID_TO_EXIT"]

    return config


def generate_save_path(config: dict) -> str:
    saving_directory = (
            Path("checkpoints")
            / config["STARTING_PHASE"]
            / config["MODEL"]
            / config["DATASET"]
            / config["LOSSCLS"]
            / config["LOSSDIV"]
            / config["LOSSKD"]
            / str(config["CLIENT_ID_TO_FORGET"])
            / str(config["Client_ID_TO_EXIT"])
            / str(config["UNLEARNING_CASE"])
            / str(config["FORGET_CLASS"])
            .replace(" ", "")
            .replace(":", "-")
            .replace(",", "_")
            .replace("{", "")
            .replace("}", "")
            / config["CONFIG_ID"]
            / f"{config['CONFIG_NUMBER']}_{config['SEED']}"
    )
    return saving_directory.as_posix()


def best_model_path_from_saving_dir(saving_dir: str) -> str:
    return (Path(saving_dir) / "models_chkpts" / "model_best.pth").as_posix()


def is_under_pretrain_dir(path: Path) -> bool:
    return "pretrain" in {p.name for p in path.parents} or path.name == "pretrain"


def find_pretrain_env_dirs(root: Path):
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        if not is_under_pretrain_dir(d):
            continue
        if ".env" in filenames or ".env.training" in filenames:
            yield d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Root directory to search")
    parser.add_argument("--out", default="best_model_paths.txt", help="Output file")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: Path does not exist: {root}")
        return

    needed = [
        "STARTING_PHASE", "MODEL", "DATASET", "LOSSCLS", "LOSSDIV", "LOSSKD",
        "CLIENT_ID_TO_FORGET", "Client_ID_TO_EXIT", "UNLEARNING_CASE",
        "FORGET_CLASS", "CONFIG_ID", "CONFIG_NUMBER", "SEED"
    ]

    hits = []
    existing_count = 0
    missing_count = 0

    for d in find_pretrain_env_dirs(root):
        cfg = load_config(d)
        if not cfg:
            continue

        missing = [k for k in needed if k not in cfg]
        if missing:
            print(f"[SKIP] {d} - missing: {missing}")
            continue

        saving_dir = generate_save_path(cfg)
        best_path = best_model_path_from_saving_dir(saving_dir)
        exists = Path(best_path).is_file()

        hits.append((d, saving_dir, best_path, exists))

        status = "✓ EXISTS" if exists else "✗ MISSING"
        if exists:
            existing_count += 1
        else:
            missing_count += 1

        print(f"\n{status}")
        print(f"  ENV: {d}")
        print(f"  PATH: {best_path}")

    # Write output
    out_path = Path(args.out).resolve()
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for dd, sv, best_path, exists in hits:
            status = "[EXISTS]" if exists else "[MISSING]"
            f.write(f"{status} {dd}:\n{best_path}\n\n")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  Total configs found: {len(hits)}")
    print(f"  Models exist: {existing_count}")
    print(f"  Models missing: {missing_count}")
    print(f"  Output written to: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()