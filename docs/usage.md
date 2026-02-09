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

Use [`sleap-label`](https://sleap.ai/guides/proofreading.html) for proofreading. For microscopy data, you can [convert TrackMate output to SLEAP format](https://gist.github.com/aaprasad/5243be0785a40e9dafa1697ce2258e3e) to use the SLEAP GUI.

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
    --no-gpu  # use CPU only
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
| `--gpu/--no-gpu` | GPU | Use GPU or CPU |
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

Run tracking on videos without ground truth:

```bash
dreem track ./data/inference \
    --checkpoint ./models/my_model.ckpt \
    --output ./results \
    --crop-size 70
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_DIR` | Path to input data directory |
| `--checkpoint` | Path to model checkpoint (`.ckpt`) |
| `--output` | Output directory for results |
| `--crop-size` | Bounding box size (should match training) |

### Common Options

```bash
dreem track ./data/inference \
    --checkpoint ./models/model.ckpt \
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
| `--max-gap` | Maximum frame gap for track continuity |
| `--anchor` | Keypoint name to use as centroid (default: `centroid`) |
| `--video-type` | Video file extension |
| `--no-gpu` | Run on CPU |

### Output

Results are saved as `.slp` files in the output directory. Open them in SLEAP or the [DREEM Visualizer](visualizer.md).

---

## Evaluation

Evaluate tracking against ground truth labels:

```bash
dreem eval ./data/test \
    --checkpoint ./models/model.ckpt \
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

## CLI Reference

Get help for any command:

```bash
dreem --help
dreem train --help
dreem track --help
dreem eval --help
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
