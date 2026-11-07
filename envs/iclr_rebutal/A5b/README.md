# A5b: No post-training, same budget (CIFAR-10)

**Question.** Is the gain from post-training only that it trains for more rounds?

**Asked by.** Dgg5, Ynse (budget-matched version of A5).

## What runs

MAX/MIN cycles for all 20 rounds (T=10), no post-training. Same total budget as A0.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `NUM_ROUNDS` | `20` |
| `LAST_MAX_STEPS` | `20` |
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`
(10 rounds of MAX/MIN cycles, then 10 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
Divergence variants: KL and JS.

## How to read it

If A0 beats A5b, the post-training stage helps beyond extra rounds.

**Notes.** The learning-rate schedule is stretched over 20 rounds.

## Layout and how to run

```
A5b/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A5b/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A5b/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A5b_cifar10.txt`.
