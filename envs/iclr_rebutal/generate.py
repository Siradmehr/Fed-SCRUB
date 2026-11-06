"""Generate the ICLR rebuttal (ablation) env folders for CIFAR-10.

Run from the repo root:   python envs/iclr_rebutal/generate.py

Layout written next to this file:

    <ROW>/README.md
    <ROW>/client_level/<backdoor|confuse|privacy>/N_5|N_10/[KL|JS/]{.env,.env.training}
    <ROW>/data_level/<backdoor|confuse|privacy>/N_5|N_10/<2|6|10>/[KL|JS/]{.env,.env.training}
    manifest.csv   (CONFIG_ID -> row / level / scenario / clients / percent / divergence)

Every leaf gets its own CONFIG_ID (20001 and up), checked against every CONFIG_ID that
any env in git history used, so no run can overwrite an earlier checkpoint. IDs follow
the order of ORDER, so append new rows at the end to keep existing IDs stable.

Each leaf folder is a complete EXP_ENV_DIR. The base values (data split, forget
settings, epochs, rounds, learning rate) are copied from the original paper envs
in git history (commit cdaf79e~1, envs/Cifar10/ICLR), so only the ablation knobs
differ between rows. The RESUME path of the pretrained model is rebuilt with
generate_save_path(), so it always matches the pretraining run's checkpoint dir.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.utils import generate_save_path, load_config  # noqa: E402

HIST = "cdaf79e~1"
HIST_BASE = "envs/Cifar10/ICLR"
SCENARIOS = ["backdoor", "confuse", "privacy"]
# (level, number of clients, percent removed). Client-level has no percentage.
SETTINGS = (
    [("client_level", n, None) for n in (5, 10)]
    + [("data_level", n, p) for n in (5, 10) for p in (2, 6, 10)]
)
PRETRAIN_CKPT = "model_latest.pth"
FIRST_CONFIG_ID = 20001  # every generated env gets its own id, counting up from here


def setting_dir(n: int, pct) -> str:
    return f"N_{n}" + (f"/{pct}" if pct else "")


def history_config_ids() -> set:
    """All CONFIG_IDs used by any env in git history (all datasets)."""
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", HIST], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    ids = set()
    for f in out:
        if f.endswith("/.env"):
            for line in git_show(f).splitlines():
                if line.startswith("CONFIG_ID="):
                    ids.add(line.split("=", 1)[1].strip())
    return ids


def git_show(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{HIST}:{path}"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout


def parse(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            d[k] = v
    return d


def dump(d: dict) -> str:
    return "\n".join(f"{k}={v}" for k, v in d.items()) + "\n"


def resume_overrides() -> dict:
    """Real checkpoint paths, if envs/iclr_rebutal/resume_paths.csv exists.

    Columns: level,scenario,clients,percent,resume  (percent empty for client_level).
    An entry replaces the path that is otherwise derived from the pretraining env.
    """
    f = OUT / "resume_paths.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8", newline="") as fh:
        return {(r["level"], r["scenario"], int(r["clients"]), str(r["percent"] or "")): r["resume"].strip()
                for r in csv.DictReader(fh)}


def pretrain_resume(level: str, scenario: str, n: int, pct) -> str:
    """Checkpoint path of the pretrained model, built the same way the server does."""
    override = resume_overrides().get((level, scenario, n, str(pct or "")))
    if override:
        return override
    src =f"{HIST_BASE}/{level}/{scenario}/pretrain/{setting_dir(n, pct)}"
    with tempfile.TemporaryDirectory() as tmp:
        for name in (".env", ".env.training"):
            Path(tmp, name).write_text(git_show(f"{src}/{name}"), encoding="utf-8")
        cfg = load_config(tmp)
    path = generate_save_path(cfg).replace("\\", "/")
    return f"{path}/models_chkpts/{PRETRAIN_CKPT}"


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
             "NUM_ROUNDS": "200", "LAST_MAX_STEPS": "0"},
        train={},
        what="Federated training from random initialization on the retained data only "
             "(forget data and, at client level, the removed client are excluded). "
             "200 rounds x 10 local epochs, the same recipe as the pretraining.",
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
        what="MAX/MIN cycles for T=5 (10 rounds), then 10 post-training rounds that keep the MIN loss: "
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
        what="Start directly in post-training for all 20 rounds. There is no MAX step, so the forget "
             "data is never used. The loss is the same as A0's post-training.",
        read="If A1 matches A0, the maximization step adds nothing on top of retain fine-tuning.",
        notes="Learning-rate schedule is plain cosine (milestones are only used when STARTING_PHASE=MAX).",
    ),
    "A2": dict(
        div=None,
        title="Fine-tune on retained data, CE only",
        question="Is plain fine-tuning on the retained data enough?",
        reviewers="Dgg5 (is it the objective, or the extra retention and recovery steps?).",
        env={"STARTING_PHASE": '"LEARN"', "TEACHER": '"NONE"', "LAST_MAX_STEPS": "0"},
        train={},
        what="20 rounds of LEARN: cross-entropy on the retain data only, no teacher, no forget data.",
        read="The simplest control. Beating A2 is what shows the forgetting objective matters.",
        notes="Uses LOCAL_EPOCHS (10) per round and gradient clipping, unlike the MIN procedure. "
              "Learning-rate schedule is plain cosine.",
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
        env={"NUM_ROUNDS": "10", "LAST_MAX_STEPS": "10"},
        train={"ALPHA": "0.5", "GAMMA": "1.0"},
        what="Stop after the 10 MAX/MIN rounds (T=5). The budget is 10 rounds, half of A0's.",
        read="Direct removal of the stage. Compare A5b for the same-budget version.",
        notes="Fewer rounds than A0 by design.",
    ),
    "A5b": dict(
        div=("KL", "JS"),
        title="No post-training, same budget",
        question="Is the gain from post-training only that it trains for more rounds?",
        reviewers="Dgg5, Ynse (budget-matched version of A5).",
        env={"NUM_ROUNDS": "20", "LAST_MAX_STEPS": "20"},
        train={"ALPHA": "0.5", "GAMMA": "1.0"},
        what="MAX/MIN cycles for all 20 rounds (T=10), no post-training. Same total budget as A0.",
        read="If A0 beats A5b, the post-training stage helps beyond extra rounds.",
        notes="The learning-rate schedule is stretched over 20 rounds.",
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
        env={"NUM_ROUNDS": "10", "LAST_MAX_STEPS": "10"},
        train={"ALPHA": "0.0", "GAMMA": "0.0"},
        what="5 MAX rounds with inert MIN rounds in between (alpha=gamma=0), no post-training.",
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


def build_row(row: str, spec: dict, next_id: list, used_ids: set, manifest: list) -> int:
    count = 0
    for level, n, pct in SETTINGS:
        for sc in SCENARIOS:
            src = f"{HIST_BASE}/{level}/{sc}/unlearn/{setting_dir(n, pct)}"
            base_env = parse(git_show(f"{src}/.env"))
            base_train = parse(git_show(f"{src}/.env.training"))
            resume = pretrain_resume(level, sc, n, pct)
            for div in (spec["div"] or (None,)):
                while str(next_id[0]) in used_ids:  # skip ids an earlier env used
                    next_id[0] += 1
                cid = str(next_id[0])
                next_id[0] += 1
                env = dict(base_env)
                env.update(spec["env"])
                if "RESUME" not in spec["env"]:
                    env["RESUME"] = resume
                env["CONFIG_ID"] = cid
                env["CONFIG_NUMBER"] = "1"
                train = dict(base_train)
                train.update(spec["train"])
                if div:
                    train["LOSSDIV"] = div
                    train["LOSSKD"] = div
                leaf = OUT / row / level / sc / setting_dir(n, pct)
                if div:
                    leaf = leaf / div
                write_leaf(leaf, env, train)
                manifest.append((cid, row, level, sc, n, pct or "", div or "",
                                 leaf.relative_to(ROOT).as_posix()))
                count += 1
    return count


def readme(row: str, spec: dict) -> str:
    rows = {**spec["env"], **spec["train"]}
    if spec["div"]:
        rows["LOSSDIV / LOSSKD"] = " or ".join(spec["div"])
    table = "\n".join(f"| `{k}` | `{v}` |" for k, v in rows.items()) or "| (none) | |"
    div_note = ("KL and JS" if spec["div"] and len(spec["div"]) == 2
                else "KL only" if spec["div"] else "not applicable")
    return f"""# {row}: {spec['title']}

**Question.** {spec['question']}

**Asked by.** {spec['reviewers']}

## What runs

{spec['what']}

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
{table}

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario ({PRETRAIN_CKPT}).
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
export EXP_ENV_DIR=envs/iclr_rebutal/{row}/data_level/backdoor/N_5/2{'/KL' if spec['div'] else ''}
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
"""


def top_readme(id_ranges: dict) -> str:
    lines = [
        "# ICLR rebuttal envs (CIFAR-10)",
        "",
        "One folder per experiment. Each has a README, then `client_level/` (N=5, N=10) and `data_level/`",
        "(N=5 and N=10, each with 2%, 6% and 10% removed), each with `backdoor`, `confuse` and `privacy`.",
        "Leaf folders are complete `EXP_ENV_DIR`s. `manifest.csv` maps every CONFIG_ID to its folder.",
        "Regenerate with `python envs/iclr_rebutal/generate.py` (needs the git history).",
        "",
        "| Row | What | Divergence | CONFIG_IDs |",
        "|---|---|---|---|",
    ]
    for r in ORDER:
        s = ROWS[r]
        d = "+".join(s["div"]) if s["div"] else "-"
        lines.append(f"| [{r}]({r}/README.md) | {s['title']} | {d} | {id_ranges[r]} |")
    lines += [
        "",
        "Priority (from the reviews): R, A7, A0, A2, A1, A3, A4, A5 first; then A5b, A6, A7b, A8.",
        "Within a row start with client_level N_5 and data_level N_5 / 2%, then the other settings.",
        "",
        "Before running: copy the pretrained checkpoints to the `RESUME` paths in the envs",
        "(they are the paths the pretraining runs write to).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    used = history_config_ids()
    next_id, manifest, ranges = [FIRST_CONFIG_ID], [], {}
    total = 0
    for row in ORDER:
        spec = ROWS[row]
        if (OUT / row).exists():
            shutil.rmtree(OUT / row)
        first = next_id[0]
        count = build_row(row, spec, next_id, used, manifest)
        ranges[row] = f"{first}-{next_id[0] - 1}"
        (OUT / row / "README.md").write_text(readme(row, spec), encoding="utf-8", newline="\n")
        total += count
        print(f"{row}: {count} env folders, CONFIG_ID {ranges[row]}")
    (OUT / "README.md").write_text(top_readme(ranges), encoding="utf-8", newline="\n")
    with open(OUT / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "row", "level", "scenario", "clients", "percent", "divergence", "env_dir"])
        w.writerows(manifest)
    print("total env folders:", total)


if __name__ == "__main__":
    main()
