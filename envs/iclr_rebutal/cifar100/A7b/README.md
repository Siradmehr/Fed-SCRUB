# A7b: CE-only post-training, gamma=1 (CIFAR-100)

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

Base: the config of the paper's unlearning run of the same setting (`../reference_configs_cifar100.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=30`, `LAST_MAX_STEPS=2`
(2 rounds of MAX/MIN cycles, then 28 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
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
export EXP_ENV_DIR=envs/iclr_rebutal/cifar100/A7b/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A7b_cifar100.txt`.
