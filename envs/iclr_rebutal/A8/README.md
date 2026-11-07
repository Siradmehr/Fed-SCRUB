# A8: Max-only, nothing else (CIFAR-10)

**Question.** What happens with the maximization alone (unbounded objective)?

**Asked by.** MQWK (the maximization is not bounded and may wreck the model).

## What runs

5 MAX rounds with inert MIN rounds in between (alpha=gamma=0), no post-training.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `NUM_ROUNDS` | `10` |
| `LAST_MAX_STEPS` | `10` |
| `ALPHA` | `0.0` |
| `GAMMA` | `0.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`
(10 rounds of MAX/MIN cycles, then 10 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
Divergence variants: KL and JS.

## How to read it

Shows the collapse case. Report accuracy per round. The learning-rate and epoch sweeps (sensitivity section) cover the rest of the stability question.

**Notes.** Relies on DISABLE_BN=true, same as A6.

## Layout and how to run

```
A8/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A8/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A8/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A8_cifar10.txt`.
