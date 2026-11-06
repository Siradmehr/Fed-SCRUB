# A1: No forgetting step: retain-only with teacher alignment and CE

**Question.** Do the gains come from the forget-set maximization, or only from the extra retain training?

**Asked by.** Dgg5, Ynse (remove forget-set divergence maximization).

## What runs

Start directly in post-training for all 20 rounds. There is no MAX step, so the forget data is never used. The loss is the same as A0's post-training.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `STARTING_PHASE` | `"POST"` |
| `LAST_MAX_STEPS` | `0` |
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `POST_PHASE` | `POST` |
| `POST_ALPHA` | `0.5` |
| `POST_GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: KL only.

## How to read it

If A1 matches A0, the maximization step adds nothing on top of retain fine-tuning.

**Notes.** Learning-rate schedule is plain cosine (milestones are only used when STARTING_PHASE=MAX).

## Layout and how to run

```
A1/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A1/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A1/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
