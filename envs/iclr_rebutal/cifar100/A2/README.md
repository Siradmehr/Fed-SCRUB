# A2: Fine-tune on retained data, CE only (CIFAR-100)

**Question.** Is plain fine-tuning on the retained data enough?

**Asked by.** Dgg5 (is it the objective, or the extra retention and recovery steps?).

## What runs

30 rounds of LEARN: cross-entropy on the retain data only, no teacher, no forget data.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `STARTING_PHASE` | `"LEARN"` |
| `TEACHER` | `"NONE"` |
| `LAST_MAX_STEPS` | `0` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs_cifar100.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=30`, `LAST_MAX_STEPS=2`
(2 rounds of MAX/MIN cycles, then 28 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
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
export EXP_ENV_DIR=envs/iclr_rebutal/cifar100/A2/data_level/backdoor/N_5/2
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A2_cifar100.txt`.
