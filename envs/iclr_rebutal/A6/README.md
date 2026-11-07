# A6: Max-only cycles, then post-training (CIFAR-10)

**Question.** Does the alternation (MAX then MIN) matter, or is MAX followed by recovery enough?

**Asked by.** MQWK and Ynse (what is different from SCRUB-style alternation), Dgg5.

## What runs

MIN rounds are made inert with alpha=gamma=0 (zero loss, Adam takes no step, clients return the weights they received). Post-training is the paper's.

## Settings that differ from the base unlearning env

| Key | Value |
|---|---|
| `ALPHA` | `0.0` |
| `GAMMA` | `0.0` |
| `POST_PHASE` | `POST` |
| `POST_ALPHA` | `0.5` |
| `POST_GAMMA` | `1.0` |
| `LOSSDIV / LOSSKD` | `KL or JS` |

Base: the config of the paper's unlearning run of the same setting (`../reference_configs.json`):
`STARTING_PHASE="MAX"`, `TEACHER="INIT"`, `NUM_ROUNDS=20`, `LAST_MAX_STEPS=10`
(10 rounds of MAX/MIN cycles, then 10 post-training rounds), `ALPHA=0.5`, `GAMMA=1.0`.
`RESUME` is the pretrained checkpoint that run started from.
Divergence variants: KL and JS.

## How to read it

Drop relative to A0 = contribution of the MIN step inside the cycles.

**Notes.** The idle MIN rounds still use up rounds of the budget. Relies on DISABLE_BN=true (with batch norm on, the running statistics would still change).

## Layout and how to run

```
A6/client_level/<backdoor|confuse|privacy>/<N_5|N_10>/<KL|JS>/
A6/data_level/<backdoor|confuse|privacy>/<N_5|N_10>/<2|6|10>/<KL|JS>/
```

Each leaf folder holds `.env` and `.env.training` and is a complete env:

```bash
export EXP_ENV_DIR=envs/iclr_rebutal/A6/data_level/backdoor/N_5/2/KL
python -m simulate
```

Every leaf has its own `CONFIG_ID` (prefix 9119, see `manifest.csv`) and `CONFIG_NUMBER=1`. None of the ids was
used by an earlier run, so runs never overwrite earlier checkpoints. Order of the runs: `../run_lists/A6_cifar10.txt`.
