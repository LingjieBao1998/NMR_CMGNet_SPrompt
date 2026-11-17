# Historical training / evaluation launchers

The 56 `run_*.sh` files here are the original experiment launchers, kept as a
record of exactly which checkpoint, dataset, seed and decode mode produced each
result.  They are **not** part of the maintained pipeline.

* Many hard-code the original environment (`/home/lingjiebao/...`,
  `/mnt/data2/...`) and GPU ids.
* Some are inference/evaluation runs (`run_external_*`) rather than training.
* The maintained entry points are `scripts/train/train_cmgnet_sprompt.sh`,
  `scripts/train/train_xgb_cress.sh` and `scripts/infer/*.sh`.

Typical historical pattern:

```bash
# CMGNet fine-tune
bash run_1025_cmgnet.sh

# external-test generation with a fixed checkpoint/gate
bash run_external_1209extrain1ep_beams10_s0_s2006.sh
```

To reuse one, copy it next to the maintained scripts, replace the interpreter
and paths, and check that the checkpoint/data names still exist.
