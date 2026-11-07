# A7: Current code: CE-only post-training, gamma=0.5 (CIFAR-10)

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

Base: the config of the paper's unlearning run of the same setting (`../reference_configs.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`
(10 rounds of MAX/MIN cycles, then 10 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
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

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A7_cifar10.txt`.
