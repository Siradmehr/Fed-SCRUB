# A2: Fine-tune on retained data, CE only

**Question.** Is plain fine-tuning on the retained data enough?

**Asked by.** Dgg5 (is it the objective, or the extra retention and recovery steps?).

## What runs

20 rounds of LEARN: cross-entropy on the retain data only, no teacher, no forget data.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `STARTING_PHASE` | `"LEARN"` |
| `TEACHER` | `"NONE"` |
| `LAST_MAX_STEPS` | `0` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: not applicable.

## How to read it

The simplest control. Beating A2 is what shows the forgetting objective matters.

**Notes.** Uses LOCAL_EPOCHS (10) per round and gradient clipping, unlike the MIN procedure. Learning-rate schedule is plain cosine.

## Layout and how to run

```
A2/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/
A2/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A2/data_level/backdoor/N_5/2
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
