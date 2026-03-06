# Fix Microscopy Demo Colab (Issue #162)

## Issues to fix
- [x] `custom_segmented_path` undefined when using Option 2 (sample data) - now `segmented_path` derived from `data_path` in cell 14
- [x] `gpu` parameter: `run_cellpose_segmentation` expects bool but was getting string - now uses `torch.cuda.is_available()` directly
- [x] Add Colab metadata to default to GPU runtime - added `accelerator: GPU` and `colab.gpuType: T4`
- [x] Support TIFF stacks (single multi-page TIFF) in addition to TIFF directories - cell 10 now detects file type
- [x] Fix output path handling - results use `os.path.abspath("./results")`
- [x] Simplify path handling - all paths derived from `data_path` in one cell (cell 14)
- [x] Results output should show full path - prints full path after tracking
- [x] Updated docs/Examples/microscopy-demo.md to match notebook changes
- [x] Fixed Colab badge link to point to `main` branch
