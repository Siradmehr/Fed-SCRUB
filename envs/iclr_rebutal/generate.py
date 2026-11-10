"""Generate the ICLR rebuttal (ablation) env folders for CIFAR-10 and CIFAR-100.

Run from the repo root:

    python envs/iclr_rebutal/generate.py --index data/run_index/runs_index.json   # first time / refresh
    python envs/iclr_rebutal/generate.py                                           # reuse the reference files
    python envs/iclr_rebutal/generate.py --dataset cifar100                        # one dataset only

Layout (CIFAR-10 at the top level, CIFAR-100 in envs/iclr_rebutal/cifar100/ with the same structure):

    <ROW>/README.md
    <ROW>/client_level/<backdoor|confuse|privacy>/N_5|N_10/[KL|JS/]{.env,.env.training}
    <ROW>/data_level/<backdoor|confuse|privacy>/N_5|N_10/<2|6|10>/[KL|JS/]{.env,.env.training}
    manifest.csv            CONFIG_ID -> row / level / scenario / clients / percent / divergence
    reference_configs*.json the past runs every env is copied from (see below)
    run_lists/<ROW>_<dataset>.txt   env paths of one row, in the order we should run them
    run_lists/run_order_<dataset>.txt   all envs, global priority order

Where the base config comes from. runs_index.json (written by index_runs.py on the server) holds
the config of every past run. For each setting (level, scenario, clients, percent) we take the
paper's unlearning run (STARTING_PHASE=MAX, KL, teacher INIT) and copy its full config, so data
split, forget settings, batch sizes, epochs, learning rate and the RESUME path are exactly what
that run used. The RESUME path is checked against the pretraining run in the index (the file must
exist there). Only the ablation knobs of each row are changed on top. The retrain row R takes its
recipe (rounds, local epochs, learning rate, batch size) from the pretraining run instead.
Percentages or client counts printed in the paper do not matter: the past run is the truth.

Every leaf gets its own CONFIG_ID: 9119, one digit for the dataset (0 = CIFAR-10, 1 = CIFAR-100)
and a 3-digit counter (91190001, ..., 91191001, ...). Filter on the prefix 9119 to pick out the
rebuttal runs; no id was used by an earlier run, so no run can overwrite an earlier checkpoint.
IDs follow the order of ORDER, so append new rows at the end to keep existing IDs stable.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.utils import load_config  # noqa: E402

HIST = "cdaf79e~1"  # git history is only used to collect CONFIG_IDs of old envs
SCENARIOS = ["backdoor", "confuse", "privacy"]
# (level, number of clients, percent removed). Client-level has no percentage.
SETTINGS = (
    [("client_level", n, None) for n in (5, 10)]
    + [("data_level", n, p) for n in (5, 10) for p in (2, 6, 10)]
)
COSINE_ROWS = ("R", "A1", "A2")  # rows without the milestone schedule (plain cosine)
COSINE_LR = "0.001"
BATCH = "512"  # one batch size for all rows and both datasets (past runs used 256 in places)
BATCH_KEYS = ("RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH")
ID_PREFIX = "9119"  # CONFIG_ID = ID_PREFIX + dataset digit + 3-digit counter
# Per dataset: model, the budget of the paper's unlearning run (B rounds, the first L of them
# MAX/MIN cycles), the number of rounds the reference run may have used (others are set to B).
DATASETS = {
    "cifar10": dict(name="CIFAR-10", model="ResNet18_small", B=20, L=10, ref_rounds=(20,),
                    digit="0", milestone_lr="0.05", out=OUT, ref_file=OUT / "reference_configs.json", path_prefix="envs/iclr_rebutal"),
    "cifar100": dict(name="CIFAR-100", model="ResNet18", B=30, L=2, ref_rounds=(30, 50),
                     digit="1", out=OUT / "cifar100", ref_file=OUT / "reference_configs_cifar100.json",
                     path_prefix="envs/iclr_rebutal/cifar100"),
}
# Keys that live in .env.training; all others go to .env.
TRAIN_KEYS = {
    "MOMENTUM", "WEIGHT_DECAY", "LR", "LR_DECAY_RATE", "SGDA_WEIGHT_DECAY", "SGDA_MOMENTUM",
    "DEVICE", "LOSSCLS", "LOSSCLS_ARGS", "LOSSKD", "LOSSKD_ARGS", "LOSSDIV", "LOSSDIV_ARGS",
    "GAMMA", "ALPHA", "BETA", "SMOOTHING", "KD_T", "DISTILL",
    "RETRAIN_BATCH", "FORGET_BATCH", "VAL_BATCH", "TEST_BATCH",
}
GENERATED_KEYS = {  # derived by the server at start-up, not part of an env
    "SAVING_DIR", "CUSTOM_CONFIG_PATH", "INITIALIZATION_SOURCE", "INITIAL_CHECKPOINT",
}


def setting_dir(n: int, pct) -> str:
    return f"N_{n}" + (f"/{pct}" if pct else "")


def norm_path(p: str) -> str:
    p = p.replace(chr(92), "/")
    return p[2:] if p.startswith("./") else p


# ------------------------------------------------- reference runs (index) ----
def git_history_ids() -> list:
    """CONFIG_IDs used by any env in git history (all datasets)."""
    try:
        files = subprocess.run(["git", "ls-tree", "-r", "--name-only", HIST], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        return []
    ids = set()
    for f in files:
        if f.endswith("/.env"):
            text = subprocess.run(["git", "show", f"{HIST}:{f}"], cwd=ROOT,
                                  capture_output=True, text=True).stdout
            for line in text.splitlines():
                if line.startswith("CONFIG_ID="):
                    ids.add(line.split("=", 1)[1].strip())
    return sorted(ids)


def in_dataset(r: dict, key: str) -> bool:
    c = r["config"]
    return str(c.get("DATASET")).strip('"').lower() == key and c.get("MODEL") == DATASETS[key]["model"]


def pick_reference(ds, key, level, sc, n, pct, runs, pretrain_by_dir):
    """The paper's unlearning run for one setting (its pretraining run must be in the index)."""
    cands = []
    for r in runs:
        c = r["config"]
        if r["phase"] != "MAX" or r["level"] != level or r["scenario"] != sc:
            continue
        if str(r["clients"]) != str(n) or str(r["percent"]) != str(pct or ""):
            continue
        if r["divergence"] not in ("KL", "JS") or str(c.get("TEACHER")).strip('"') != "INIT":
            continue
        if int(str(c.get("NUM_ROUNDS"))) not in ds["ref_rounds"] or int(c.get("LAST_MAX_STEPS", 0)) != ds["L"]:
            continue
        if not isinstance(c.get("FORGET_CLASS"), dict) or str(c.get("CONFIG_ID")) == "67":
            continue
        res = str(c.get("RESUME") or "")
        pre = pretrain_by_dir.get(norm_path(res).split("/models_chkpts")[0]) if res else None
        which = "best_model" if Path(res).name == "model_best.pth" else "latest_model"
        if pre is None or not pre.get(which):
            continue  # the pretrained checkpoint it resumed from is not in the index
        cands.append(r)
    if not cands:
        raise SystemExit(f"[{key}] no reference unlearning run for {level}/{sc}/N={n}/pct={pct}")

    def sig(r):  # identical configs (apart from where they were saved) count as one
        return json.dumps({k: v for k, v in r["config"].items() if k not in GENERATED_KEYS}, sort_keys=True)

    if any(r["divergence"] == "KL" for r in cands):  # prefer KL; JS only when no KL run exists
        cands = [r for r in cands if r["divergence"] == "KL"]
    best_rounds = min(int(str(r["config"]["NUM_ROUNDS"])) for r in cands)  # prefer the first listed budget
    for pref in ds["ref_rounds"]:
        if any(int(str(r["config"]["NUM_ROUNDS"])) == pref for r in cands):
            best_rounds = pref
            break
    cands = [r for r in cands if int(str(r["config"]["NUM_ROUNDS"])) == best_rounds]
    distinct = {}
    for r in sorted(cands, key=lambda r: r["run_dir"]):
        distinct.setdefault(sig(r), r)
    chosen = list(distinct.values())[0]
    if len(distinct) > 1:
        print(f"  note: {len(distinct)} different reference configs for {level}/{sc}/N={n}/pct={pct}; "
              f"using {chosen['run_dir']}")
    return chosen


def build_reference(key: str, index_file: Path) -> dict:
    ds = DATASETS[key]
    everything = json.loads(index_file.read_text(encoding="utf-8"))
    runs = [r for r in everything if in_dataset(r, key)]
    pretrain_by_dir = {norm_path(r["run_dir"]): r for r in runs if r["phase"] == "PRETRAIN"}
    settings = {}
    for level, n, pct in SETTINGS:
        for sc in SCENARIOS:
            r = pick_reference(ds, key, level, sc, n, pct, runs, pretrain_by_dir)
            cfg, notes = dict(r["config"]), []
            if r["divergence"] != "KL":  # no KL run in the index: copy the JS run, switch the divergence to KL
                cfg["LOSSDIV"] = cfg["LOSSKD"] = "KL"
                notes.append(f"copied from a {r['divergence']} run, LOSSDIV/LOSSKD set to KL")
            if int(str(cfg["NUM_ROUNDS"])) != ds["B"]:
                notes.append(f"reference run had {cfg['NUM_ROUNDS']} rounds, NUM_ROUNDS set to {ds['B']}")
                cfg["NUM_ROUNDS"] = str(ds["B"])
            for note in notes:
                print(f"  note: [{key}] {level}/{sc}/N={n}/pct={pct}: {note}")
            pre_dir = norm_path(str(cfg["RESUME"])).split("/models_chkpts")[0]
            settings[f"{level}/{sc}/{n}/{pct or ''}"] = {
                "source_run": r["run_dir"], "note": "; ".join(notes), "config": cfg,
                "pretrain_run": pre_dir, "pretrain_config": pretrain_by_dir[pre_dir]["config"],
            }
    used = sorted({str(r["config"].get("CONFIG_ID")) for r in everything} | set(git_history_ids()))
    return {"index": index_file.as_posix(), "used_config_ids": used, "settings": settings}


def fmt(v) -> str:
    """Python value from custom_config.json -> text of an env file."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}:{x}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def split_config(cfg: dict):
    env, train = {}, {}
    for k, v in cfg.items():
        if k in GENERATED_KEYS:
            continue
        (train if k in TRAIN_KEYS else env)[k] = fmt(v)
    return env, train


def dump(d: dict) -> str:
    return "\n".join(f"{k}={v}" for k, v in d.items()) + "\n"


def canon(v):
    return json.loads(json.dumps(v))  # dict keys -> str, tuples -> lists


def fill(x, ds):
    """Replace the budget tokens <B> <L> <T> <P> <BT> in the text of a row spec."""
    if isinstance(x, str):
        for tok, val in (("<BT>", ds["B"] // 2), ("<B>", ds["B"]), ("<L>", ds["L"]),
                         ("<T>", ds["L"] // 2), ("<P>", ds["B"] - ds["L"])):
            x = x.replace(tok, str(val))
        return x
    if isinstance(x, dict):
        return {k: fill(v, ds) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(fill(v, ds) for v in x)
    return x


# ---------------------------------------------------------------- rows ----
# env: overrides for .env, train: overrides for .env.training.
# div: which divergences to generate ("KL", "JS"), or None when it does not matter.
POST_FULL = {"POST_PHASE": "POST", "POST_ALPHA": "0.5", "POST_GAMMA": "1.0"}
ROWS = {
    "R": dict(
        div=None,
        title="Retrain reference (exact retraining)",
        question="How far is each method from retraining on the retained data from scratch?",
        reviewers="Dgg5 (Eq. 4 targets the retrained model), MQWK (golden retraining is not reported).",
        env={"STARTING_PHASE": '"EXACT"', "RESUME": "", "TEACHER": '"NONE"',
             "NUM_ROUNDS": "@pre:NUM_ROUNDS", "LOCAL_EPOCHS": "@pre:LOCAL_EPOCHS", "LAST_MAX_STEPS": "0"},
        train={"LR": "@pre:LR", "RETRAIN_BATCH": "@pre:RETRAIN_BATCH"},
        what="Federated training from random initialization on the retained data only "
             "(forget data and, at client level, the removed client are excluded). "
             "Rounds and local epochs are those of the pretraining run of the same setting. The learning rate is 0.001 with a plain cosine schedule (0.5 is too high for Adam), and the batch size is 512.",
        read="This is the reference, not an ablation. Report every other row's gap to it "
             "(accuracy, ASR / confused accuracy, MIA). For MIA the target is this model's own MIA, not 50%.",
        notes="Does not use the pretrained model, so no RESUME. Needs ~the same time as pretraining. "
              "The retained set is the same for all three scenarios, so the three runs differ only in the "
              "evaluation they log.",
    ),
    "A0": dict(
        div=("KL", "JS"),
        title="Full method, paper version",
        question="What does the method as described in Algorithm 1 give?",
        reviewers="Dgg5, Ynse (the reference row of the ablation).",
        env={}, train={"ALPHA": "0.5", "GAMMA": "1.0", **POST_FULL},
        what="MAX/MIN cycles for T=<T> (<L> rounds), then <P> post-training rounds that keep the MIN loss: "
             "gamma * CE + alpha * divergence(student, teacher), with gamma=1, alpha=0.5.",
        read="All component-removal rows (A1-A6) are compared with this row.",
        notes="Uses the POST phase (MIN procedure with the loss weights sent by the server).",
    ),
    "A1": dict(
        div=("KL",),
        title="No forgetting step: retain-only with teacher alignment and CE",
        question="Do the gains come from the forget-set maximization, or only from the extra retain training?",
        reviewers="Dgg5, Ynse (remove forget-set divergence maximization).",
        env={"STARTING_PHASE": '"POST"', "LAST_MAX_STEPS": "0"},
        train={"ALPHA": "0.5", "GAMMA": "1.0", **POST_FULL},
        what="Start directly in post-training for all <B> rounds. There is no MAX step, so the forget "
             "data is never used. The loss is the same as A0's post-training.",
        read="If A1 matches A0, the maximization step adds nothing on top of retain fine-tuning.",
        notes="Learning-rate schedule is plain cosine from LR 0.001 (milestones are only used when STARTING_PHASE=MAX).",
    ),
    "A2": dict(
        div=None,
        title="Fine-tune on retained data, CE only",
        question="Is plain fine-tuning on the retained data enough?",
        reviewers="Dgg5 (is it the objective, or the extra retention and recovery steps?).",
        env={"STARTING_PHASE": '"LEARN"', "TEACHER": '"NONE"', "LAST_MAX_STEPS": "0"},
        train={},
        what="<B> rounds of LEARN: cross-entropy on the retain data only, no teacher, no forget data.",
        read="The simplest control. Beating A2 is what shows the forgetting objective matters.",
        notes="Uses LOCAL_EPOCHS (10) per round and gradient clipping, unlike the MIN procedure. "
              "Learning-rate schedule is plain cosine from LR 0.001.",
    ),
    "A3": dict(
        div=("KL", "JS"),
        title="No teacher-alignment term on the retain data",
        question="Is the retain-side divergence term needed?",
        reviewers="Dgg5 (remove the retention divergence term), Ynse.",
        env={}, train={"ALPHA": "0.0", "GAMMA": "1.0", **{**POST_FULL, "POST_ALPHA": "0.0"}},
        what="alpha=0 in both the MIN rounds and the post-training: the retain loss is CE only. "
             "The MAX step is unchanged.",
        read="Drop relative to A0 = contribution of the teacher-alignment term.",
        notes="",
    ),
    "A4": dict(
        div=("KL", "JS"),
        title="No supervised retain loss",
        question="Is the supervised CE term on the retain data needed?",
        reviewers="Dgg5 (remove the supervised retention loss), Ynse.",
        env={}, train={"ALPHA": "0.5", "GAMMA": "0.0", **{**POST_FULL, "POST_GAMMA": "0.0"}},
        what="gamma=0 in both the MIN rounds and the post-training: the retain loss is the "
             "divergence to the teacher only.",
        read="Drop relative to A0 = contribution of the supervised loss.",
        notes="",
    ),
    "A5": dict(
        div=("KL", "JS"),
        title="No post-training",
        question="Does the post-training recovery stage matter?",
        reviewers="Dgg5, Ynse (remove the post-training recovery stage).",
        env={"NUM_ROUNDS": "<L>", "LAST_MAX_STEPS": "<L>"},
        train={"ALPHA": "0.5", "GAMMA": "1.0"},
        what="Stop after the <L> MAX/MIN rounds (T=<T>). The budget is <L> rounds instead of A0's <B>.",
        read="Direct removal of the stage. Compare A5b for the same-budget version.",
        notes="Fewer rounds than A0 by design.",
    ),
    "A5b": dict(
        div=("KL", "JS"),
        title="No post-training, same budget",
        question="Is the gain from post-training only that it trains for more rounds?",
        reviewers="Dgg5, Ynse (budget-matched version of A5).",
        env={"NUM_ROUNDS": "<B>", "LAST_MAX_STEPS": "<B>"},
        train={"ALPHA": "0.5", "GAMMA": "1.0"},
        what="MAX/MIN cycles for all <B> rounds (T=<BT>), no post-training. Same total budget as A0.",
        read="If A0 beats A5b, the post-training stage helps beyond extra rounds.",
        notes="The learning-rate schedule is stretched over <B> rounds.",
    ),
    "A6": dict(
        div=("KL", "JS"),
        title="Max-only cycles, then post-training",
        question="Does the alternation (MAX then MIN) matter, or is MAX followed by recovery enough?",
        reviewers="MQWK and Ynse (what is different from SCRUB-style alternation), Dgg5.",
        env={}, train={"ALPHA": "0.0", "GAMMA": "0.0", **POST_FULL},
        what="MIN rounds are made inert with alpha=gamma=0 (zero loss, Adam takes no step, "
             "clients return the weights they received). Post-training is the paper's.",
        read="Drop relative to A0 = contribution of the MIN step inside the cycles.",
        notes="The idle MIN rounds still use up rounds of the budget. Relies on DISABLE_BN=true "
              "(with batch norm on, the running statistics would still change).",
    ),
    "A7": dict(
        div=("KL", "JS"),
        title="Current code: CE-only post-training, gamma=0.5",
        question="What do the numbers in the paper correspond to?",
        reviewers="Anchor for all rows (reproduces the reported setup).",
        env={}, train={"ALPHA": "0.5", "GAMMA": "0.5", "POST_PHASE": "LEARN"},
        what="Original behavior before the ALPHA/GAMMA forwarding fix: gamma=0.5 in the MIN rounds, "
             "alpha=0.5, and post-training is LEARN (CE only on the retain data).",
        read="Should reproduce the paper's numbers (client-level results need the aggregation fix, "
             "so they will differ slightly from the old ones).",
        notes="",
    ),
    "A7b": dict(
        div=("KL", "JS"),
        title="CE-only post-training, gamma=1",
        question="Separates the effect of gamma from the effect of the post-training type (A0 vs A7).",
        reviewers="Supports reading A0 vs A7.",
        env={}, train={"ALPHA": "0.5", "GAMMA": "1.0", "POST_PHASE": "LEARN"},
        what="Same as A7 but gamma=1 in the MIN rounds.",
        read="A7b vs A7 = effect of gamma. A0 vs A7b = effect of the post-training type.",
        notes="Optional.",
    ),
    "A8": dict(
        div=("KL", "JS"),
        title="Max-only, nothing else",
        question="What happens with the maximization alone (unbounded objective)?",
        reviewers="MQWK (the maximization is not bounded and may wreck the model).",
        env={"NUM_ROUNDS": "<L>", "LAST_MAX_STEPS": "<L>"},
        train={"ALPHA": "0.0", "GAMMA": "0.0"},
        what="<T> MAX rounds with inert MIN rounds in between (alpha=gamma=0), no post-training.",
        read="Shows the collapse case. Report accuracy per round. The learning-rate and epoch "
             "sweeps (sensitivity section) cover the rest of the stability question.",
        notes="Relies on DISABLE_BN=true, same as A6.",
    ),
}

ORDER = ["R", "A0", "A1", "A2", "A3", "A4", "A5", "A5b", "A6", "A7", "A7b", "A8"]


def write_leaf(path: Path, env: dict, train: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text(dump(env), encoding="utf-8", newline="\n")
    (path / ".env.training").write_text(dump(train), encoding="utf-8", newline="\n")


def from_pretrain(overrides: dict, pre_cfg: dict) -> dict:
    """Resolve '@pre:KEY' values to the pretraining run's value of KEY."""
    return {k: (fmt(pre_cfg[v[5:]]) if isinstance(v, str) and v.startswith("@pre:") else v)
            for k, v in overrides.items()}


def build_row(row, spec, ds, next_id, used_ids, manifest, refs) -> int:
    count = 0
    for level, n, pct in SETTINGS:
        for sc in SCENARIOS:
            entry = refs[f"{level}/{sc}/{n}/{pct or ''}"]
            ref = entry["config"]
            base_env, base_train = split_config(ref)
            env_over = from_pretrain(spec["env"], entry["pretrain_config"])
            train_over = from_pretrain(spec["train"], entry["pretrain_config"])
            if row in COSINE_ROWS:  # plain cosine from 0.001: 0.5 with Adam collapses the model
                train_over["LR"] = COSINE_LR
            elif ds.get("milestone_lr"):  # milestone-then-cosine rows start at 0.05 (0.5 with Adam is too high)
                train_over["LR"] = ds["milestone_lr"]
            train_over.update({k: BATCH for k in BATCH_KEYS})  # batch size 512 in every env
            for div in (spec["div"] or (None,)):
                cid = f"{ID_PREFIX}{ds['digit']}{next_id[0]:03d}"
                next_id[0] += 1
                if cid in used_ids:
                    raise SystemExit(f"CONFIG_ID {cid} was used by an earlier run")
                env = dict(base_env)
                env.update(env_over)
                env["CONFIG_ID"] = cid
                env["CONFIG_NUMBER"] = "1"
                train = dict(base_train)
                train.update(train_over)
                if div:
                    train["LOSSDIV"] = div
                    train["LOSSKD"] = div
                leaf = ds["out"] / row / level / sc / setting_dir(n, pct)
                if div:
                    leaf = leaf / div
                write_leaf(leaf, env, train)
                check_leaf(leaf, ref, {**env_over, **train_over}, div)
                manifest.append((cid, row, level, sc, n, pct or "", div or "",
                                 leaf.relative_to(ROOT).as_posix()))
                count += 1
    return count


def check_leaf(leaf: Path, ref: dict, overrides: dict, div) -> None:
    """The env must load, and differ from its reference run only where the row says so."""
    cfg = load_config(str(leaf))
    allowed = set(overrides) | {"CONFIG_ID", "CONFIG_NUMBER"}
    if div:
        allowed |= {"LOSSDIV", "LOSSKD"}
    bad = [k for k in sorted(set(cfg) | set(ref))
           if k not in GENERATED_KEYS | allowed | {"Resume_Training"}
           and canon(cfg.get(k)) != canon(ref.get(k)) and str(cfg.get(k)) != str(ref.get(k))]
    if bad:
        raise SystemExit(f"{leaf}: differs from its reference run in {bad}")


def readme(row: str, spec: dict, ds: dict, key: str) -> str:
    rows = {**spec["env"], **spec["train"]}
    rows = {k: ("(as in the pretraining run)" if str(v).startswith("@pre:") else v) for k, v in rows.items()}
    if spec["div"]:
        rows["LOSSDIV / LOSSKD"] = " or ".join(spec["div"])
    table = "\n".join(f"| `{k}` | `{v}` |" for k, v in rows.items()) or "| (none) | |"
    div_note = ("KL and JS" if spec["div"] and len(spec["div"]) == 2
                else "KL only" if spec["div"] else "not applicable")
    pp = ds["path_prefix"]
    return f"""# {row}: {spec['title']} ({ds['name']})

**Question.** {spec['question']}

**Asked by.** {spec['reviewers']}

## What runs

{spec['what']}

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
{table}

Base: the config of the paper's unlearning run of the same setting (`../reference_configs{'' if key == 'cifar10' else '_cifar100'}.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS={ds['B']}`, `LAST_MAX_STEPS={ds['L']}`
({ds['L']} rounds of MAX/MIN cycles, then {ds['B'] - ds['L']} post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
Divergence variants: {div_note}.

## How to read it

{spec['read']}

{('**Notes.** ' + spec['notes']) if spec['notes'] else ''}

## Layout and how to run

```
{row}/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/{'<KL|JS>/' if spec['div'] else ''}
{row}/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/{'<KL|JS>/' if spec['div'] else ''}
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR={pp}/{row}/data_level/backdoor/N_5/2{'/KL' if spec['div'] else ''}
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/{row}_{key}.txt`.
"""


def top_readme(id_ranges: dict, ds: dict, key: str) -> str:
    lines = [
        f"# ICLR rebuttal envs ({ds['name']})",
        "",
        "One folder per experiment. Each has a README, then `client_level/` (N=5, N=10) and `data_level/`",
        "(N=5 and N=10, each with 2%, 6% and 10% removed), each with `backdoor`, `confuse` and `privacy`.",
        "Leaf folders are complete `EXP_ENV_DIR`s. `manifest.csv` maps every CONFIG_ID to its folder.",
        "Regenerate with `python envs/iclr_rebutal/generate.py` (reads `reference_configs*.json`; add",
        "`--index <runs_index.json>` to rebuild them from the server's run index).",
        "Settings applied to every env: batch size 512; learning rate 0.001 with plain cosine for the rows",
        "without MAX/MIN cycles (R, A1, A2)" + ("; milestone-then-cosine rows start at 0.05." if ds.get("milestone_lr")
                                                 else "; milestone-then-cosine rows keep the LR of the paper's run."),
        "Every CONFIG_ID starts with `9119` (`91190xxx` CIFAR-10, `91191xxx` CIFAR-100): filter on that prefix",
        "to pick out these runs, e.g. the `config_id` column of the `index_runs.py` output.",
        "",
        "| Row | What | Divergence | CONFIG_IDs |",
        "|---|---|---|---|",
    ]
    for r in ORDER:
        s = fill(ROWS[r], ds)
        d = "+".join(s["div"]) if s["div"] else "-"
        lines.append(f"| [{r}]({r}/README.md) | {s['title']} | {d} | {id_ranges[r]} |")
    lines += [
        "",
        f"Run order: `{'' if key == 'cifar10' else '../'}run_lists/run_order_{key}.txt` (all envs, by priority) and",
        f"`{'' if key == 'cifar10' else '../'}run_lists/<ROW>_{key}.txt` (one row, by priority).",
        "Run a list (skips comment and blank lines):",
        "",
        "```bash",
        f"grep -v '^\s*#' envs/iclr_rebutal/run_lists/run_order_{key}.txt | grep -v '^\s*$' | while read d; do",
        "  EXP_ENV_DIR=$d python -m simulate",
        "done",
        "```",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------ run order ----
def setting_group(level, n, pct, sc) -> int:
    """0 = first settings (5 clients; data 2% / client level; backdoor, confuse), 1 = their privacy runs,
    2 = other 5-client data settings, 3 = 10 clients."""
    core = (level == "client_level" and n == 5) or (level == "data_level" and n == 5 and pct == 2)
    if core:
        return 1 if sc == "privacy" else 0
    return 2 if n == 5 else 3


def row_sort_key(m):
    _, row, level, sc, n, pct, div, _ = m
    return (setting_group(level, int(n), int(pct) if pct != "" else None, sc),
            {"KL": 0, "": 0, "JS": 1}[div], 0 if level == "data_level" else 1,
            SCENARIOS.index(sc), int(n), int(pct) if pct != "" else 0)


# (title, rows, groups, divergences)
STAGES = [
    ("1. core ablation, KL, backdoor + confuse (5 clients: data 2%, client level)",
     ["A7", "A0", "A2", "A1", "A3", "A4", "A5"], {0}, ("KL", "")),
    ("2. same settings with JS", ["A0", "A7", "A3", "A4", "A5"], {0}, ("JS",)),
    ("3. privacy on the same settings", ["A7", "A0", "A2", "A5"], {1}, ("KL", "")),
    ("4. privacy, remaining ablation rows", ["A1", "A3", "A4"], {1}, ("KL", "")),
    ("5. budget-matched and max-only rows on the first settings (KL)", ["A5b", "A6", "A7b", "A8"], {0, 1}, ("KL", "")),
    ("6. budget-matched and max-only rows on the first settings (JS)", ["A5b", "A6", "A7b", "A8"], {0, 1}, ("JS",)),
]


def write_run_lists(key: str, manifest: list) -> None:
    out = OUT / "run_lists"
    out.mkdir(exist_ok=True)
    for row in ORDER:
        items = sorted((m for m in manifest if m[1] == row), key=row_sort_key)
        (out / f"{row}_{key}.txt").write_text(
            "".join(m[7] + "\n" for m in items), encoding="utf-8", newline="\n")
    done, lines = set(), [f"# {DATASETS[key]['name']}: all envs in the order we should run them.",
                          "# Lines starting with # are comments (skip them when looping).",
                          f"# Row R (retrain) is not listed here, see R_{key}.txt.", ""]
    for title, rows, groups, divs in STAGES:
        chosen = [m for r in rows for m in sorted((m for m in manifest if m[1] == r), key=row_sort_key)
                  if setting_group(m[2], int(m[4]), int(m[5]) if m[5] != "" else None, m[3]) in groups
                  and m[6] in divs and m[0] not in done]
        lines.append(f"# --- {title}: {len(chosen)} runs ---")
        lines += [m[7] for m in chosen]
        lines.append("")
        done |= {m[0] for m in chosen}
    rest = sorted((m for m in manifest if m[0] not in done and m[1] != "R"),  # R is run separately
                  key=lambda m: (row_sort_key(m)[0], ORDER.index(m[1])) + row_sort_key(m)[1:])
    lines.append(f"# --- 7. everything else (other settings, by setting group then row): {len(rest)} runs ---")
    lines += [m[7] for m in rest]
    (out / f"run_order_{key}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="runs_index.json from index_runs.py; (re)builds the reference files")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = ap.parse_args()
    for key in (DATASETS if args.dataset == "all" else [args.dataset]):
        ds = DATASETS[key]
        print(f"=== {ds['name']}")
        if args.index:
            ds["ref_file"].write_text(json.dumps(build_reference(key, Path(args.index)), indent=1),
                                      encoding="utf-8", newline="\n")
            print("wrote", ds["ref_file"].relative_to(ROOT))
        if not ds["ref_file"].exists():
            raise SystemExit(f"{ds['ref_file'].name} missing: run once with --index data/run_index/runs_index.json")
        data = json.loads(ds["ref_file"].read_text(encoding="utf-8"))
        refs, used = data["settings"], set(data["used_config_ids"])
        next_id, manifest, ranges, total = [1], [], {}, 0
        for row in ORDER:
            spec = fill(ROWS[row], ds)
            if (ds["out"] / row).exists():
                shutil.rmtree(ds["out"] / row)
            first = next_id[0]
            count = build_row(row, spec, ds, next_id, used, manifest, refs)
            ranges[row] = (f"{ID_PREFIX}{ds['digit']}{first:03d}-{ID_PREFIX}{ds['digit']}{next_id[0] - 1:03d}")
            (ds["out"] / row / "README.md").write_text(readme(row, spec, ds, key), encoding="utf-8", newline="\n")
            total += count
            print(f"{row}: {count} env folders, CONFIG_ID {ranges[row]}")
        (ds["out"] / "README.md").write_text(top_readme(ranges, ds, key), encoding="utf-8", newline="\n")
        with open(ds["out"] / "manifest.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config_id", "row", "level", "scenario", "clients", "percent", "divergence", "env_dir"])
            w.writerows(manifest)
        write_run_lists(key, manifest)
        print("total env folders:", total)


if __name__ == "__main__":
    main()
