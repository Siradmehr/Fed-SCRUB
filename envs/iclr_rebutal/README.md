# ICLR rebuttal envs (CIFAR-10)

One folder per experiment. Each has a README, then `client_level/` (N=5, N=10) and `data_level/`
(N=5 and N=10, each with 2%, 6% and 10% removed), each with `backdoor`, `confuse` and `privacy`.
Leaf folders are complete `EXP_ENV_DIR`s. `manifest.csv` maps every CONFIG_ID to its folder.
Regenerate with `python envs/iclr_rebutal/generate.py` (needs the git history).

| Row | What | Divergence | CONFIG_IDs |
|---|---|---|---|
| [R](R/README.md) | Retrain reference (exact retraining) | - | 20001-20024 |
| [A0](A0/README.md) | Full method, paper version | KL+JS | 20025-20072 |
| [A1](A1/README.md) | No forgetting step: retain-only with teacher alignment and CE | KL | 20073-20096 |
| [A2](A2/README.md) | Fine-tune on retained data, CE only | - | 20097-20120 |
| [A3](A3/README.md) | No teacher-alignment term on the retain data | KL+JS | 20121-20168 |
| [A4](A4/README.md) | No supervised retain loss | KL+JS | 20169-20216 |
| [A5](A5/README.md) | No post-training | KL+JS | 20217-20264 |
| [A5b](A5b/README.md) | No post-training, same budget | KL+JS | 20265-20313 |
| [A6](A6/README.md) | Max-only cycles, then post-training | KL+JS | 20314-20361 |
| [A7](A7/README.md) | Current code: CE-only post-training, gamma=0.5 | KL+JS | 20362-20409 |
| [A7b](A7b/README.md) | CE-only post-training, gamma=1 | KL+JS | 20410-20457 |
| [A8](A8/README.md) | Max-only, nothing else | KL+JS | 20458-20505 |

Priority (from the reviews): R, A7, A0, A2, A1, A3, A4, A5 first; then A5b, A6, A7b, A8.
Within a row start with client_level N_5 and data_level N_5 / 2%, then the other settings.

Before running: copy the pretrained checkpoints to the `RESUME` paths in the envs
(they are the paths the pretraining runs write to).
