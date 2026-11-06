# A4: No supervised retain loss

**Question.** Is the supervised CE term on the retain data needed?

**Asked by.** Dgg5 (remove the supervised retention loss), Ynse.

## What runs

gamma=0 in both the MIN rounds and the post-training: the retain loss is the divergence to the teacher only.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `ALPHA` | `0.5` |
| `GAMMA` | `0.0` |
| `POST_PHASE` | `POST` |
| `POST_ALPHA` | `0.5` |
| `POST_GAMMA` | `0.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: KL and JS.

## How to read it

Drop relative to A0 = contribution of the supervised loss.



## Layout and how to run

```
A4/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A4/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A4/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
