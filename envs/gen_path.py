import os
import re
from pathlib import Path

def parse_env_file(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        v = re.split(r"\s+#", v, maxsplit=1)[0].strip()

        if (len(v) >= 2) and (v[0] == v[-1]) and v[0] in ("'", '"'):
            v = v[1:-1]

        env[k] = v

    return env


def load_config_from_dir(d: Path) -> dict:
    cfg = {}
    cfg.update(parse_env_file(d / ".env"))
    cfg.update(parse_env_file(d / ".env.training"))  # override
    return cfg


def canonicalize_keys(cfg: dict) -> dict:
    """
    Make key naming consistent with generate_save_path().
    - Accepts CLIENT_ID_TO_EXIT or Client_ID_TO_EXIT, but uses Client_ID_TO_EXIT.
    """
    if "Client_ID_TO_EXIT" not in cfg and "CLIENT_ID_TO_EXIT" in cfg:
        cfg["Client_ID_TO_EXIT"] = cfg["CLIENT_ID_TO_EXIT"]
    return cfg
from typing import Dict
import os
import dotenv
INT_KEYS = [
    "RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH",
    "NUM_CLASSES", "LOCAL_EPOCHS", "MIN_EPOCHS", "MAX_EPOCHS", "LAST_MAX_STEPS"
]
from dotenv import dotenv_values
from ast import literal_eval
def load_config(path: str = "./envs") -> Dict:
    """Load and process configuration from environment files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration directory not found: {path}")

    # Load configuration files
    env_path = os.path.join(path, ".env")
    training_path = os.path.join(path, ".env.training")


    if not os.path.exists(env_path) or not os.path.exists(training_path):
        raise FileNotFoundError(f"Required configuration files missing in {path}")

    config = {
        **dotenv_values(env_path),
        **dotenv_values(training_path),
    }

    # Process configuration values
    # Handle forget class
    if "FORGET_CLASS" in config:
        config["FORGET_CLASS"] = literal_eval(config["FORGET_CLASS"])

    if "MAP_CONFUSE" in config:
        config["MAP_CONFUSE"] = literal_eval(config["MAP_CONFUSE"])

    # Convert integer keys
    for key in INT_KEYS:
        if key in config:
            config[key] = int(config[key])

    # Process comma-separated integer lists
    for key in ["CLIENT_ID_TO_FORGET", "LR_ROUND", "Client_ID_TO_EXIT"]:
        if config[key]:
            config[key] = [int(i) for i in str(config[key]).split(",")]
        else:
            config[key] = []


    return config

from pathlib import Path

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
    return saving_directory.as_posix()   # <-- forces / separators


def best_model_path_from_saving_dir(saving_dir: str) -> str:
    return (Path(saving_dir) / "models_chkpts" / "model_best.pth").as_posix()
#
#
# def generate_save_path(config: dict) -> str:
#     saving_directory = os.path.join(
#         "./checkpoints",
#         config["STARTING_PHASE"],
#         config["MODEL"],
#         config["DATASET"],
#         config["LOSSCLS"],
#         config["LOSSDIV"],
#         config["LOSSKD"],
#         str(config["CLIENT_ID_TO_FORGET"]),
#         str(config["Client_ID_TO_EXIT"]),
#         str(config["UNLEARNING_CASE"]),
#         str(config["FORGET_CLASS"])
#             .replace(" ", "")
#             .replace(":", "-")
#             .replace(",", "_")
#             .replace("{", "")
#             .replace("}", ""),
#         config["CONFIG_ID"],
#         f"{config['CONFIG_NUMBER']}_{config['SEED']}",
#     )
#     return saving_directory
#
#
# def best_model_path_from_saving_dir(saving_dir: str) -> str:
#     return os.path.join(saving_dir, "models_chkpts", "model_best.pth")


def is_under_pretrain_dir(path: Path) -> bool:
    return "pretrain" in {p.name for p in path.parents} or path.name == "pretrain"


def find_pretrain_env_dirs(root: Path):
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        if not is_under_pretrain_dir(d):
            continue
        if ".env" in filenames or ".env.training" in filenames:
            yield d


from pathlib import Path

def main(
    root="envs",
    create_dirs=False,
    print_only_existing_best=False,
    out_file="best_model_paths.txt",
):
    root = Path(root)

    needed = [
        "STARTING_PHASE", "MODEL", "DATASET", "LOSSCLS", "LOSSDIV", "LOSSKD",
        "CLIENT_ID_TO_FORGET", "Client_ID_TO_EXIT", "UNLEARNING_CASE",
        "FORGET_CLASS", "CONFIG_ID", "CONFIG_NUMBER", "SEED"
    ]

    hits = []
    for d in find_pretrain_env_dirs(root):
        cfg = load_config(d)
        if not cfg:
            continue

        missing = [k for k in needed if k not in cfg]
        if missing:
            print(f"[SKIP] {d} missing keys: {missing}")
            continue

        saving_dir = generate_save_path(cfg)                      # POSIX string
        best_path = best_model_path_from_saving_dir(saving_dir)   # POSIX string
        exists = Path(best_path).is_file()                        # robust with /

        if print_only_existing_best and not exists:
            continue

        hits.append((d, saving_dir, best_path, exists))

        print(f"\nENV DIR:   {d}")
        print(f"SAVING:    {saving_dir}")
        print(f"BEST PATH: {best_path}")
        print(f"EXISTS:    {exists}")

        if create_dirs:
            Path(saving_dir).mkdir(parents=True, exist_ok=True)

    # ---- write output file (one best_path per line) ----
    out_path = Path(out_file).resolve()
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for _, _, best_path, exists in hits:
            if print_only_existing_best and not exists:
                continue
            f.write(f"{d}:\n")
            f.write(best_path + "\n\n\n")

    print(f"\nDone. Found {len(hits)} pretrain env dir(s) matching requirements.")
    print(f"Wrote {len(hits)} path(s) to: {out_path.as_posix()}")
    return hits


# Example usage:
main(
    root="f-fumMnist",
    create_dirs=False,
    print_only_existing_best=False,   # set True to only store existing model_best.pth
    out_file="best_model_paths.txt",
)

# If you want ONLY the ones that actually have model_best.pth:
# main(root="f-fumMnist", create_dirs=False, print_only_existing_best=True)
