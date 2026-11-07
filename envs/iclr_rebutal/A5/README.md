# A5: No post-training (CIFAR-10)

**Question.** Does the post-training recovery stage matter?

**Asked by.** Dgg5, Ynse (remove the post-training recovery stage).

## What runs

Stop after the 10 MAX/MIN rounds (T=5). The budget is 10 rounds instead of A0's 20.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `NUM_ROUNDS` | `10` |
| `LAST_MAX_STEPS` | `10` |
| `ALPHA` | `0.5` |
| `GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`
(10 rounds of MAX/MIN cycles, then 10 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
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

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A5_cifar10.txt`.
