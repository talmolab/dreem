# PR #174 Review Fixes

## Tasks

- [x] Fix 1: Rename `_to_frame_array()` → `load_frames()` and make public
- [x] Fix 2: `setup_ctc_dirs()` — point to existing dir instead of copying
- [x] Fix 3: `run_tracking()` — accept pre-built CTC paths (`ctc_paths` param)
- [x] Fix 4: Add type annotations to all new function signatures
- [x] Fix 5: Grayscale conversion — add `grayscale` kwarg to `load_frames()`
- [x] Fix 6: Validate frame/mask shape agreement in `setup_ctc_dirs()`
- [x] Fix 7: Update stale module docstring in `run_cellpose_segmentation.py`
- [x] Fix 8: Export `run_tracking` from `dreem/inference/__init__.py`

## Verification

- [x] `uv run ruff check` — all checks passed
- [x] `uv run ruff format --check` — all files formatted
- [x] `uv run pytest tests/test_cellpose_utils.py -v` — 27/27 passed
- [x] `uv run pytest tests/ -v` — 154 passed, 2 skipped, 0 failures
