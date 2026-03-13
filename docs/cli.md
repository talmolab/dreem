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
| `dreem convert` | Convert external tracking formats to `.slp` files |
| `dreem render` | Render tracking results as video (CTC masks or SLEAP labels) |

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
| `--device` | `auto` | Accelerator: `auto`, `gpu`, `cpu`, `mps` |
| `--gpu/--no-gpu`, `-g` | - | **Deprecated.** Use `--device` instead |
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
| `--checkpoint`, `-ckpt` | Yes | Model checkpoint: local path, shortname (`animals`, `microscopy`), HuggingFace repo (`org/repo`), or HuggingFace URL |
| `--output`, `-o` | Yes | Output directory |
| `--crop-size`, `-cs` | Yes | Crop size (should match training) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video-type`, `-vt` | `mp4` | Video file extension |
| `--anchor`, `-a` | `centroid` | Anchor keypoint name |
| `--clip-length`, `-cl` | `32` | Frames per batch |
| `--max-tracks`, `-mx` | - | Maximum number of tracks |
| `--confidence-threshold`, `-conf` | `0` | Threshold for flagging low-confidence predictions. Results saved to the suggested frames section of the output .slp file. |
| `--max-dist`, `-md` | - | Maximum center distance between frames |
| `--max-dist-multiplier`, `-mdm` | - | Penalty multiplier applied when center distance exceeds `--max-dist` (default: `1.0`) |
| `--max-gap`, `-mg` | - | Maximum frame gap for track continuity |
| `--iou-mode`, `-iou` | `mult` | IOU mode (`mult` or `add`) |
| `--overlap-thresh`, `-ot` | - | Overlap threshold |
| `--max-angle`, `-ma` | - | Maximum angle difference |
| `--front-node`, `-fn` | - | Front nodes for orientation (can repeat) |
| `--back-node`, `-bn` | - | Back nodes for orientation (can repeat) |
| `--slp-file`, `-slp` | - | Specific SLEAP label files (can repeat) |
| `--video-file`, `-vid` | - | Specific video files (can repeat) |
| `--output-format`, `-of` | `native` | Output format: `native` (`.tif`/`.slp`), `csv`, or `both` |
| `--render`, `-R` | - | Render tracked results to video at this path (CTC masks or SLEAP labels) |
| `--save-meta`, `-sm` | - | Save frame metadata |
| `--device` | `auto` | Accelerator: `auto`, `gpu`, `cpu`, `mps` |
| `--gpu/--no-gpu`, `-g` | - | **Deprecated.** Use `--device` instead |
| `--limit-batches` | - | Limit inference to N batches (for debugging) |
| `--config`, `-c` | - | YAML config file |
| `--set`, `-s` | - | Config overrides |
| `--quiet`, `-q` | - | Suppress progress output |
| `--verbose` | - | Enable verbose logging |

### Example

Using a pretrained shortname (auto-downloads from HuggingFace):

```bash
dreem track ./data/inference \
    --checkpoint animals \
    --output ./results \
    --crop-size 70 \
    --max-tracks 5
```

Using a local checkpoint path:

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
    --checkpoint animals \
    --output ./eval_results \
    --crop-size 70 \
    --max-tracks 5
```

### Output

- `.slp` or `.tif` files with predicted tracks (native format)
- `.csv` trajectory file (if `--output-format csv` or `both`; columns: `frame`, `detection_idx`, `track_id`, `confidence`, `centroid_x`, `centroid_y`)
- `motmetrics.csv` with evaluation metrics
- `.h5` file with detailed results

---

## Convert

Convert tracking data from external formats to SLEAP `.slp` files.

```bash
dreem convert FORMAT --labels PATH --videos PATH [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `FORMAT` | Yes | Source format to convert from (currently: `trackmate`) |
| `--labels`, `-l` | Yes | Paths to label files — repeat for multiple (e.g., `-l file1.csv -l file2.csv`) |
| `--videos`, `-v` | Yes | Paths to video files — repeat for multiple (e.g., `-v file1.tif -v file2.tif`) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `.` | Output directory for converted files |
| `--to-mp4`, `-m` | - | Convert TIF/ND2 videos to `.mp4` format |
| `--to-npy`, `-n` | - | Convert TIF videos to `.npy` format |

### Example

```bash
dreem convert trackmate \
    -l ./data/labels1.csv \
    -l ./data/labels2.csv \
    -v ./data/video1.tif \
    -v ./data/video2.tif \
    --output ./converted \
    --to-mp4
```

### Output

- `.slp` files with tracks and detections (one per label/video pair)
- `.mp4` or `.npy` video files (if `--to-mp4` or `--to-npy` is set)

1-indexed frame numbers are automatically converted to 0-indexed.

---

## Render

Render tracking results as videos with colored overlays, trajectory trails, and ID labels. Supports both CTC-format mask TIFF stacks and SLEAP `.slp` label files.

```bash
dreem render INPUT_PATH --output PATH [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `INPUT_PATH` | Yes | Input file: tracked mask TIFF (`.tif`/`.tiff`) or SLEAP labels (`.slp`) |
| `--output`, `-o` | Yes | Output video file path (e.g., `output.mp4`) |

### Options

**Common options (both formats):**

| Option | Default | Description |
|--------|---------|-------------|
| `--palette`, `-p` | `distinct` | Color palette name |
| `--trail-length` | `10` | Trail length in frames |
| `--show-ids/--no-ids` | `show-ids` | Show track ID labels |
| `--show-trails/--no-trails` | `show-trails` | Show trajectory trails |
| `--fps` | `30.0` | Output video frame rate |
| `--scale` | `1.0` | Scale factor for rendering |
| `--quiet`, `-q` | - | Suppress progress output |

**CTC mask options (`.tif`/`.tiff`):**

| Option | Default | Description |
|--------|---------|-------------|
| `--raw-frames`, `-r` | - | Raw video/TIFF frames for background (default: black) |
| `--mask-alpha` | `0.5` | Mask overlay opacity (0–1) |
| `--show-masks/--no-masks` | `show-masks` | Show mask overlays |
| `--show-centroids/--no-centroids` | `show-centroids` | Show centroid markers |

**SLEAP label options (`.slp`):**

| Option | Default | Description |
|--------|---------|-------------|
| `--line-width` | `2.0` | Skeleton edge line width |
| `--show-nodes/--no-nodes` | `show-nodes` | Show keypoint markers |
| `--show-edges/--no-edges` | `show-edges` | Show skeleton edges |

### Example

```bash
# CTC masks: render with raw frames as background
dreem render ./results/tracked.tif \
    --output ./results/visualization.mp4 \
    --raw-frames ./data/raw.tif \
    --trail-length 15 \
    --fps 15

# CTC masks: render on black background, no ID labels
dreem render ./results/tracked.tif \
    --output ./results/visualization.mp4 \
    --no-ids

# SLEAP labels: render with tracks
dreem render ./results/tracked.slp \
    --output ./results/visualization.mp4

# SLEAP labels: render without skeleton edges
dreem render ./results/tracked.slp \
    --output ./results/visualization.mp4 \
    --no-edges
```

### Auto-render after tracking

Use the `--render` flag on `dreem track` to automatically render after tracking (works with both CTC and SLEAP outputs):

```bash
dreem track ./data \
    --checkpoint microscopy \
    --output ./results \
    --crop-size 128 \
    --render ./results/video.mp4
```

### Output

- `.mp4` video file with colored overlays and annotations

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
