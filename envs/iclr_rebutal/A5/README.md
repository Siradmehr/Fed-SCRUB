# A5: No post-training

**Question.** Does the post-training recovery stage matter?

**Asked by.** Dgg5, Ynse (remove the post-training recovery stage).

## What runs

Stop after the 10 MAX/MIN rounds (T=5). The budget is 10 rounds, half of A0's.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `NUM_ROUNDS` | `10` |
| `LAST_MAX_STEPS` | `10` |
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: KL and JS.

## How to read it

Direct removal of the stage. Compare A5b for the same-budget version.

**Notes.** Fewer rounds than A0 by design.

## Layout and how to run

```
A5/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A5/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A5/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
