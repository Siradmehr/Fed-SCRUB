# A7b: CE-only post-training, gamma=1

**Question.** Separates the effect of gamma from the effect of the post-training type (A0 vs A7).

**Asked by.** Supports reading A0 vs A7.

## What runs

Same as A7 but gamma=1 in the MIN rounds.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `POST_PHASE` | `LEARN` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: KL and JS.

## How to read it

A7b vs A7 = effect of gamma. A0 vs A7b = effect of the post-training type.

**Notes.** Optional.

## Layout and how to run

```
A7b/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A7b/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A7b/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
