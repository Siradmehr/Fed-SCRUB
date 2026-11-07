# A0: Full method, paper version (CIFAR-100)

**Question.** What does the method as described in Algorithm 1 give?

**Asked by.** Dgg5, Ynse (the reference row of the ablation).

## What runs

MAX/MIN cycles for T=1 (2 rounds), then 28 post-training rounds that keep the MIN loss: gamma * CE + alpha * divergence(student, teacher), with gamma=1, alpha=0.5.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `POST_PHASE` | `POST` |
| `POST_ALPHA` | `0.5` |
| `POST_GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs_cifar100.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=30`, `LAST_MAX_STEPS=2`
(2 rounds of MAX/MIN cycles, then 28 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
Divergence variants: KL and JS.

## How to read it

All component-removal rows (A1-A6) are compared with this row.

**Notes.** Uses the POST phase (MIN procedure with the loss weights sent by the server).

## Layout and how to run

```
A0/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A0/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/cifar100/A0/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A0_cifar100.txt`.
