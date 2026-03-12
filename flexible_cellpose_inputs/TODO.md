# Issue #170: Flexible CellPose inputs and simplified CTC directory management

## Status: Complete

## Tasks

- [x] **Phase 1: Flexible `run_cellpose_segmentation()` inputs**
  - [x] Add `_to_frame_array()` helper function
  - [x] Refactor `run_cellpose_segmentation()` signature and body
  - [x] Support: directory of TIFFs, TIFF stack, video file, numpy array, sio.Video
  - [x] Optional `output_path` (return array without writing if None)

- [x] **Phase 2: CTC Directory Helpers**
  - [x] Create `dreem/utils/ctc_helpers.py` with `setup_ctc_dirs()`
  - [x] Update `dreem/utils/__init__.py` exports

- [x] **Phase 3: High-Level `run_tracking()` API**
  - [x] Add `run_tracking()` wrapper in `dreem/inference/track.py`

- [x] **Phase 4: Update Notebook & Docs**
  - [x] Simplify `examples/microscopy-demo.ipynb`
  - [x] Sync `docs/Examples/microscopy-demo.md`

- [x] **Phase 5: Dependencies**
  - [x] Add `tifffile` to `pyproject.toml` dependencies

- [x] **Tests**
  - [x] Create `tests/test_cellpose_utils.py` with all test cases (18 tests)
  - [x] Run full test suite for regression (145 passed, 2 skipped, 0 failed)

## Verification
- [x] `uv run pytest tests/test_cellpose_utils.py -v` — 18/18 passed
- [x] `uv run pytest tests/ -v` — 145 passed, 2 skipped
- [x] `uv run ruff check` — All checks passed
