# Fix: Inference config leaks training-only settings (Issue #169)

## To-do

- [x] 1. Add mode-aware `get_trainer()` to `dreem/io/config.py`
  - [x] Add `_TRAINING_ONLY_TRAINER_KEYS`, `_INFERENCE_TRAINER_DEFAULTS`, `_TRAINING_ONLY_CONFIG_SECTIONS` constants
  - [x] Add `mode` parameter to `get_trainer()`
  - [x] Strip training-only keys in inference/eval mode
  - [x] Apply inference defaults for unset keys
- [x] 2. Update inference callers
  - [x] `dreem/inference/track.py:248` -> pass `mode="inference"`
  - [x] `dreem/inference/eval.py:35` -> pass `mode="eval"`
- [x] 3. CLI updates in `dreem/cli.py`
  - [x] Add `_strip_training_config_sections()` helper
  - [x] Add `--device` flag to track/eval/train commands
  - [x] Deprecate `--gpu` flag (hidden)
  - [x] Add `--limit-batches` flag to track/eval
  - [x] Call `_strip_training_config_sections()` in inference commands
- [x] 4. Fix `create_chunks_other` in `dreem/datasets/base_dataset.py:234`
- [x] 5. Add tests in `tests/test_config.py`
- [x] 6. Update `docs/cli.md` for `--device` flag
- [x] 7. Lint and test - all 125 tests pass, 0 lint errors
