# Usage

This guide covers the DREEM workflow: preparing data, training models, running inference, and evaluating results. For a hands-on introduction, see the [Quickstart](./quickstart.md) or the [Examples](./Examples/).

## Data Preparation

### What You Need

**For training**, you need:

1. **Videos** - `.mp4`, `.avi`, or `.tif` stacks (for microscopy)
2. **Labels with ground truth tracks** - Detections with temporally consistent identity labels in [SLEAP](https://sleap.ai) (`.slp`) or [Cell Tracking Challenge](https://celltrackingchallenge.net) format

**For inference**, you need:

1. **Videos** - Same formats as above
2. **Detections only** - No ground truth tracks required; just detections in [SLEAP](https://sleap.ai) (`.slp`) or [Cell Tracking Challenge](https://celltrackingchallenge.net) format

### Getting Detections

DREEM decouples detection from tracking, so you can use any detection method. Here are some popular methods:

- **Animal tracking**: [SLEAP](https://sleap.ai) for pose estimation
- **Microscopy**: [CellPose](https://www.cellpose.org), [Ilastik](https://www.ilastik.org), or [TrackMate](https://imagej.net/plugins/trackmate/)

### Directory Structure

**SLEAP format** (animal tracking):
```
dataset/
├── train/
│   ├── video1.mp4
│   ├── video1.slp
│   ├── video2.mp4
│   └── video2.slp
├── val/
│   ├── video3.mp4
│   └── video3.slp
└── test/  # optional
    ├── video4.mp4
    └── video4.slp
```

**Cell Tracking Challenge format** (microscopy):
```
dataset/
├── train/
│   ├── seq_01/
│   │   ├── frame000.tif
│   │   └── ...
│   └── seq_01_GT/TRA/
│       ├── frame000.tif  # labeled masks
│       └── ...
└── val/
    └── ...
```

### Proofreading Ground Truth

Good tracking results require accurate ground truth. Before training:

1. Ensure **no identity switches** in your annotations
2. Verify **detection accuracy** - crops are centered on each instance

Use [`sleap-label`](https://sleap.ai/guides/proofreading.html) for proofreading. For microscopy data, you can convert TrackMate output to SLEAP format using `dreem convert` (see below) to use the SLEAP GUI.

---

## Training

Train a model with `dreem train`:

```bash
dreem train ./data/train --val-dir ./data/val --crop-size 70
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `TRAIN_DIR` | Path to training data directory |
| `--val-dir` | Path to validation data directory |
| `--crop-size` | Size of bounding box around each instance (pixels) |

### Common Options

```bash
dreem train ./data/train \
    --val-dir ./data/val \
    --crop-size 70 \
    --epochs 30 \
    --lr 0.0001 \
    --run-name my_experiment \
    --device cpu  # use CPU only
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 20 | Maximum training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--d-model` | 128 | Model embedding dimension |
| `--nhead` | 1 | Number of attention heads |
| `--encoder-layers` | 1 | Transformer encoder layers |
| `--decoder-layers` | 1 | Transformer decoder layers |
| `--clip-length` | 32 | Frames per training batch |
| `--run-name` | dreem_train | Name for logging/checkpoints |
| `--device` | `auto` | Accelerator: `auto`, `gpu`, `cpu`, `mps` |
| `--video-type` | mp4 | Video file extension (`mp4`, `tif`, etc.) |

### Using Config Files

For advanced options (augmentations, schedulers, callbacks), use a YAML config:

```bash
dreem train ./data/train --val-dir ./data/val --crop-size 70 --config ./my_config.yaml
```

See [Training Configuration](configs/training.md) for all available options.

### Output

Training saves checkpoints to `./models/{run_name}/`. The final checkpoint is named `*final*.ckpt`.

---

## Tracking (Inference)

Run tracking on videos without ground truth. You can use a pretrained model shortname (`animals` or `microscopy`) which auto-downloads from HuggingFace, or pass a local checkpoint path:

```bash
# Using a pretrained shortname (auto-downloads and caches from HuggingFace)
dreem track ./data/inference \
    --checkpoint animals \
    --output ./results \
    --crop-size 70

# Using a local checkpoint
dreem track ./data/inference \
    --checkpoint ./models/my_model.ckpt \
    --output ./results \
    --crop-size 70
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_DIR` | Path to input data directory |
| `--checkpoint` | Pretrained shortname (`animals`, `microscopy`), HuggingFace repo (`org/repo`), HuggingFace URL, or local path to a `.ckpt` file |
| `--output` | Output directory for results |
| `--crop-size` | Bounding box size (should match training) |

### Common Options

```bash
dreem track ./data/inference \
    --checkpoint animals \
    --output ./results \
    --crop-size 70 \
    --max-tracks 5 \
    --confidence-threshold 0.8
```

| Option | Description |
|--------|-------------|
| `--max-tracks` | Maximum number of tracks (set to number of animals/cells) |
| `--confidence-threshold` | Flag low-confidence predictions for review |
| `--max-dist` | Maximum center distance between frames |
| `--max-dist-multiplier` | Penalty multiplier applied when center distance exceeds `--max-dist` |
| `--max-gap` | Maximum frame gap for track continuity |
| `--anchor` | Keypoint name to use as centroid (default: `centroid`) |
| `--video-type` | Video file extension |
| `--device cpu` | Run on CPU |

!!! tip "Using `--max-dist` to enforce distance or motion based constraints"
    If you know the maximum distance the instances in your data can travel frame over frame, using `--max-dist` can help track them better. It penalizes matches that exceed the threshold you set. Using `--max-dist-multiplier` multiplies this distance penalty relative to the model's association scores, pushing tracking behavior closer to a pure distance-based tracker.

### Output

Results are saved as `.slp` files in the output directory. Open them in SLEAP or the [DREEM Visualizer](visualizer.md).

---

## Evaluation

Evaluate tracking against ground truth labels:

```bash
dreem eval ./data/test \
    --checkpoint animals \
    --output ./eval_results \
    --crop-size 70
```

This computes standard MOT metrics (MOTA, IDF1, ID switches) and saves results to the output directory.

### Options

`dreem eval` accepts the same options as `dreem track`. The input directory must contain ground truth labels.

### Output

- `.slp` files with predicted tracks
- `motmetrics.csv` with evaluation metrics
- `.h5` file with detailed results

---

## Converting External Formats

Convert tracking data from external tools (e.g., TrackMate) to SLEAP `.slp` format:

```bash
dreem convert trackmate \
    -l ./data/labels1.csv -l ./data/labels2.csv \
    -v ./data/video1.tif -v ./data/video2.tif \
    --output ./converted \
    --to-mp4
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `FORMAT` | Source format to convert from (currently: `trackmate`) |
| `--labels`, `-l` | Paths to label files (repeat for multiple files) |
| `--videos`, `-v` | Paths to video files (repeat for multiple files) |

### Common Options

```bash
dreem convert trackmate \
    -l labels.csv \
    -v video.tif \
    --output ./converted \
    --to-mp4  # convert TIF to MP4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `.` | Output directory for converted files |
| `--to-mp4`, `-m` | - | Convert TIF/ND2 videos to `.mp4` format |
| `--to-npy`, `-n` | - | Convert TIF videos to `.npy` format |

### Output

- `.slp` files with tracks and detections (one per video)
- `.mp4` or `.npy` video files (if `--to-mp4` or `--to-npy` is set)

1-indexed frame numbers from TrackMate are automatically converted to 0-indexed.

---

## CLI Reference

Get help for any command:

```bash
dreem --help
dreem train --help
dreem track --help
dreem eval --help
dreem convert --help
```

### Config Overrides

Override individual config values with `--set`:

```bash
dreem track ./data --checkpoint model.ckpt --output ./results \
    --crop-size 70 \
    --set tracker.window_size=16 \
    --set tracker.decay_time=0.9
```
For fine grained control, use a config file as mentioned above.

---

## Next Steps

- [Quickstart](./quickstart.md) - Run tracking in 5 minutes
- [End-to-end Demo](./Examples/dreem-demo.md) - Full training and evaluation workflow
- [Microscopy Demo](./Examples/microscopy-demo.md) - CellPose + DREEM pipeline
- [Configuration Reference](./configs/) - All config options
