# ICLR rebuttal envs (CIFAR-10)

One folder per experiment. Each has a README, then `client_level/` (N=5, N=10) and `data_level/`
(N=5 and N=10, each with 2%, 6% and 10% removed), each with `backdoor`, `confuse` and `privacy`.
Leaf folders are complete `EXP_ENV_DIR`s. `manifest.csv` maps every CONFIG_ID to its folder.
Regenerate with `python envs/iclr_rebutal/generate.py` (reads `reference_configs*.json`; add
`--index <runs_index.json>` to rebuild them from the server's run index).
Settings applied to every env: batch size 512; learning rate 0.001 with plain cosine for the rows
without MAX/MIN cycles (R, A1, A2); milestone-then-cosine rows start at 0.05.
Every CONFIG_ID starts with `9119` (`91190xxx` CIFAR-10, `91191xxx` CIFAR-100): filter on that prefix
to pick out these runs, e.g. the `config_id` column of the `index_runs.py` output.

| Row | What | Divergence | CONFIG_IDs |
|---|---|---|---|
| [R](R/README.md) | Retrain reference (exact retraining) | - | 91190001-91190024 |
| [A0](A0/README.md) | Full method, paper version | KL+JS | 91190025-91190072 |
| [A1](A1/README.md) | No forgetting step: retain-only with teacher alignment and CE | KL | 91190073-91190096 |
| [A2](A2/README.md) | Fine-tune on retained data, CE only | - | 91190097-91190120 |
| [A3](A3/README.md) | No teacher-alignment term on the retain data | KL+JS | 91190121-91190168 |
| [A4](A4/README.md) | No supervised retain loss | KL+JS | 91190169-91190216 |
| [A5](A5/README.md) | No post-training | KL+JS | 91190217-91190264 |
| [A5b](A5b/README.md) | No post-training, same budget | KL+JS | 91190265-91190312 |
| [A6](A6/README.md) | Max-only cycles, then post-training | KL+JS | 91190313-91190360 |
| [A7](A7/README.md) | Current code: CE-only post-training, gamma=0.5 | KL+JS | 91190361-91190408 |
| [A7b](A7b/README.md) | CE-only post-training, gamma=1 | KL+JS | 91190409-91190456 |
| [A8](A8/README.md) | Max-only, nothing else | KL+JS | 91190457-91190504 |

Run order: `run_lists/run_order_cifar10.txt` (all envs, by priority) and
`run_lists/<ROW>_cifar10.txt` (one row, by priority).
Run a list (skips comment and blank lines):

```bash
grep -v '^\s*#' envs/iclr_rebutal/run_lists/run_order_cifar10.txt | grep -v '^\s*$' | while read d; do
  EXP_ENV_DIR=$d python -m simulate
done
```
