# Command-line Interface

DREEM provides a unified CLI for training, tracking, and evaluation.

```bash
dreem --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `dreem train` | Train a DREEM model |
| `dreem track` | Run tracking inference (no ground truth) |
| `dreem eval` | Evaluate tracking against ground truth |

---

## Training

Train a model on your dataset.

```bash
dreem train TRAIN_DIR --val-dir VAL_DIR --crop-size SIZE [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `TRAIN_DIR` | Yes | Training data directory |
| `--val-dir`, `-vd` | Yes | Validation data directory |
| `--crop-size`, `-cs` | Yes | Crop size around each instance (pixels) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video-type`, `-vt` | `mp4` | Video file extension (`mp4`, `tif`, etc.) |
| `--epochs`, `-e` | `20` | Maximum training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--d-model` | `128` | Model embedding dimension |
| `--nhead` | `1` | Number of attention heads |
| `--encoder-layers` | `1` | Transformer encoder layers |
| `--decoder-layers` | `1` | Transformer decoder layers |
| `--anchor`, `-a` | `centroid` | Anchor keypoint name |
| `--clip-length`, `-cl` | `32` | Frames per training batch |
| `--gpu/--no-gpu`, `-g` | GPU | Use GPU or CPU |
| `--config`, `-c` | - | YAML config file for advanced options |
| `--logger`, `-l` | - | Logger type (e.g., `WandbLogger`) |
| `--run-name`, `-rn` | `dreem_train` | Run name for logging/checkpoints |
| `--set`, `-s` | - | Config overrides (e.g., `--set model.nhead=4`) |
| `--quiet`, `-q` | - | Suppress progress output |
| `--verbose` | - | Enable verbose logging |

### Example

```bash
dreem train ./data/train \
    --val-dir ./data/val \
    --crop-size 70 \
    --epochs 30 \
    --run-name my_experiment
```

---

## Tracking

Run tracking inference on videos without ground truth labels.

```bash
dreem track INPUT_DIR --checkpoint PATH --output DIR --crop-size SIZE [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `INPUT_DIR` | Yes | Input data directory |
| `--checkpoint`, `-ckpt` | Yes | Model checkpoint path (`.ckpt`) |
| `--output`, `-o` | Yes | Output directory |
| `--crop-size`, `-cs` | Yes | Crop size (should match training) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video-type`, `-vt` | `mp4` | Video file extension |
| `--anchor`, `-a` | `centroid` | Anchor keypoint name |
| `--clip-length`, `-cl` | `32` | Frames per batch |
| `--max-tracks`, `-mx` | - | Maximum number of tracks |
| `--confidence-threshold`, `-conf` | `0` | Threshold for flagging low-confidence predictions |
| `--max-dist`, `-md` | - | Maximum center distance between frames |
| `--max-gap`, `-mg` | - | Maximum frame gap for track continuity |
| `--iou-mode`, `-iou` | `mult` | IOU mode (`mult` or `add`) |
| `--overlap-thresh`, `-ot` | - | Overlap threshold |
| `--max-angle`, `-ma` | - | Maximum angle difference |
| `--front-node`, `-fn` | - | Front nodes for orientation (can repeat) |
| `--back-node`, `-bn` | - | Back nodes for orientation (can repeat) |
| `--slp-file`, `-slp` | - | Specific SLEAP label files (can repeat) |
| `--video-file`, `-vid` | - | Specific video files (can repeat) |
| `--save-meta`, `-sm` | - | Save frame metadata |
| `--gpu/--no-gpu`, `-g` | GPU | Use GPU or CPU |
| `--config`, `-c` | - | YAML config file |
| `--set`, `-s` | - | Config overrides |
| `--quiet`, `-q` | - | Suppress progress output |
| `--verbose` | - | Enable verbose logging |

### Example

```bash
dreem track ./data/inference \
    --checkpoint ./models/model.ckpt \
    --output ./results \
    --crop-size 70 \
    --max-tracks 5
```

---

## Evaluation

Evaluate tracking performance against ground truth labels. Computes MOT metrics (MOTA, IDF1, ID switches).

```bash
dreem eval INPUT_DIR --checkpoint PATH --output DIR --crop-size SIZE [OPTIONS]
```

### Arguments & Options

Same as `dreem track`. The input directory must contain ground truth labels.

### Example

```bash
dreem eval ./data/test \
    --checkpoint ./models/model.ckpt \
    --output ./eval_results \
    --crop-size 70 \
    --max-tracks 5
```

### Output

- `.slp` files with predicted tracks
- `motmetrics.csv` with evaluation metrics
- `.h5` file with detailed results

---

## Config Overrides

Override any config value using `--set` with dot notation:

```bash
dreem track ./data \
    --checkpoint model.ckpt \
    --output ./results \
    --crop-size 70 \
    --set tracker.window_size=16 \
    --set tracker.decay_time=0.9
```

For complex configurations, use a YAML file with `--config`:

```bash
dreem train ./data/train --val-dir ./data/val --crop-size 70 --config ./my_config.yaml
```

See [Training Config](./configs/training.md) and [Inference Config](./configs/inference.md) for all available options.
