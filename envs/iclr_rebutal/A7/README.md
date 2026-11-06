# A7: Current code: CE-only post-training, gamma=0.5

**Question.** What do the numbers in the paper correspond to?

**Asked by.** Anchor for all rows (reproduces the reported setup).

## What runs

Original behavior before the ALPHA/GAMMA forwarding fix: gamma=0.5 in the MIN rounds, alpha=0.5, and post-training is LEARN (CE only on the retain data).

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `ALPHA` | `0.5` |
| `GAMMA` | `0.5` |
| `POST_PHASE` | `LEARN` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: KL and JS.

## How to read it

Should reproduce the paper's numbers (client-level results need the aggregation fix, so they will differ slightly from the old ones).



## Layout and how to run

```
A7/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A7/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A7/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
