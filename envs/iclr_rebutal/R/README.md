# R: Retrain reference (exact retraining)

**Question.** How far is each method from retraining on the retained data from scratch?

**Asked by.** Dgg5 (Eq. 4 targets the retrained model), MQWK (golden retraining is not reported).

## What runs

Federated training from random initialization on the retained data only (forget data and, at client level, the removed client are excluded). 200 rounds x 10 local epochs, the same recipe as the pretraining.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `STARTING_PHASE` | `"EXACT"` |
| `RESUME` | `` |
| `TEACHER` | `"NONE"` |
| `NUM_ROUNDS` | `200` |
| `LAST_MAX_STEPS` | `0` |

Base (from the paper's CIFAR-10 envs): `STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `LOCAL_EPOCHS=10`,
`MIN_EPOCHS=1`, `MAX_EPOCHS=1`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`, `SEED=17`, `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` points to the pretrained checkpoint of the same scenario (model_latest.pth).
Divergence variants: not applicable.

## How to read it

This is the reference, not an ablation. Report every other row's gap to it (accuracy, ASR / confused accuracy, MIA). For MIA the target is this model's own MIA, not 50%.

**Notes.** Does not use the pretrained model, so no RESUME. Needs ~the same time as pretraining. The retained set is the same for all three scenarios, so the three runs differ only in the evaluation they log.

## Layout and how to run

```
R/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/
R/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/R/data_level/backdoor/N_5/2
python -m simulate
```

Every leaf has its own `CONFIG_ID` (see `../manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was used by an
earlier env, so runs never overwrite earlier checkpoints.
