# Inference Configuration

This guide describes the parameters used for inference configuration. Most parameters can be set via CLI flags (see `dreem track --help` or `dreem eval --help`) or through a YAML config file.

## Required Parameters

* `ckpt_path` (`str`): Path to the model checkpoint file.
* `outdir` (`str`): Directory where tracking results will be saved.

## Dataset Configuration

The `dataset.test_dataset` section configures the input data for inference.

### Directory-based Input (Recommended)

Use the `dir` section to automatically discover videos and labels in a directory:

* `path` (`str`): Path to directory containing videos and labels (use absolute paths).
* `labels_suffix` (`str`): File extension for label files (e.g., `.slp`, `.csv`, `.xml`).
* `vid_suffix` (`str`): File extension for video files (e.g., `.mp4`, `.avi`, `.tif`, `.tiff`).

### File-based Input

Alternatively, specify files explicitly:

* `slp_files` (`list[str]`): List of paths to SLEAP label files (`.slp`).
* `video_files` (`list[str]`): List of paths to video files.

### Dataset Parameters

* `crop_size` (`int`): Size (in pixels) of the square bounding box around each instance. Should match the approximate size of your tracked objects.
* `clip_length` (`int`): Number of frames per chunk when processing videos (default: 32).
* `chunk` (`bool`): Whether to chunk videos into smaller clips (default: `true`).
* `anchors` (`str` | `list[str]` | `int`): 
  * String: Single node name to center crops around (e.g., `"centroid"`).
  * List: Multiple node names to use as crop centers.
  * Integer: Number of anchors to randomly select.
* `padding` (`int`): Amount of padding added to each side of the bounding box (default: 0).
* `dilation_radius_px` (`int`): Size of mask around keypoint (pixels) to mask out background.
* `max_detection_overlap` (`float`): IOU threshold above which detections are considered duplicates.
* `detection_iou_threshold` (`bool`): Whether to use IOU threshold for detection filtering.

## Tracker Configuration

The `tracker` section controls tracking behavior and post-processing.

### Core Tracking Parameters

* `max_tracks` (`int`): Maximum number of tracks that can be created. Set this to the number of objects you expect to track.
* `overlap_thresh` (`float`): Trajectory overlap threshold for assignment (default: 0.01).
* `max_center_dist` (`float`): Maximum distance (pixels) an instance can move between frames to be considered the same track.
* `max_gap` (`int`): Maximum number of frames a trajectory can be missing before termination.
* `confidence_threshold` (`float`): Threshold below which frames are flagged as potential errors (default: 0).

### IOU and Threshold Settings

* `iou` (`str` | `None`): IOU reweighting mode. Options: `"mult"` (multiplicative), `"max"`, or `None`.
* `mult_thresh` (`bool`): Whether to use multiplicative threshold weighting.

### Orientation and Angle Constraints

* `max_angle_diff` (`float`): Maximum angle difference (degrees) allowed for track assignment.
* `front_nodes` (`list[str]`): List of node names defining the front of the object for orientation.
* `back_nodes` (`list[str]`): List of node names defining the back of the object for orientation.
* `angle_diff_penalty_multiplier` (`float`): Multiplier for angle difference penalty (default: 1.0).
* `distance_penalty_multiplier` (`float`): Multiplier for distance penalty (default: 2.0).

### Advanced Parameters

* `decay_time` (`float` | `None`): Weight for temporal decay in post-processing. Set to `null` to disable.

## Dataloader Configuration

* `test_dataloader.num_workers` (`int`): Number of subprocesses for data loading. Use `0` for single-process loading.
* `test_dataloader.shuffle` (`bool`): Whether to shuffle data (typically `false` for inference).

## Trainer Configuration

* `trainer.accelerator` (`str`): Device to use. Options: `"gpu"`, `"cuda"`, or `"cpu"`.
* `trainer.devices` (`list[int]`): List of device indices (e.g., `[0]` for first GPU).
* `trainer.limit_test_batches` (`float`): Fraction of test batches to process (default: 1.0).

## Other Parameters

* `save_frame_meta` (`bool`): Whether to save frame-level metadata (default: `false`).
* `dataset.metrics.test` (`str` | `list[str]`): Metrics to compute during evaluation. Use `"all"` for all metrics.

## Example Configuration

```YAML
# str: Path to the model checkpoint file
ckpt_path: <str>

# str: Directory where tracking results will be saved
outdir: <str>

dataset:
  test_dataset:
    # str: Node name to center crops around
    anchors: <str>
    
    # int: Number of frames per chunk when processing videos (default: 32)
    clip_length: <int>
    
    # int: Size (in pixels) of the square bounding box around each instance
    #      Should match approximate size of tracked objects
    crop_size: <int>
    
    # float: IOU threshold for detection filtering (default: 0; no filtering)
    max_detection_overlap: <float>
    
    # int: Size of mask around keypoint (in pixels) to mask out background (default: 0; no masking)
    dilation_radius_px: <int>
    
    # Directory-based input (recommended)
    dir:
      # str: File extension for label files (e.g., ".slp", ".tif", ".tiff")
      labels_suffix: <str>
      
      # str: Path to directory containing videos and labels (use absolute paths)
      path: <str>
      
      # str: File extension for video files (e.g., ".mp4", ".avi", ".tif", ".tiff")
      vid_suffix: <str>
    
    # int: Amount of padding added to each side of the bounding box (default: 0)
    padding: <int>

# bool: Whether to save frame-level metadata (default: false)
save_frame_meta: <bool>

tracker:
  # float | null: Maximum angle difference (degrees) allowed for track assignment (optional)
  max_angle_diff: <float | null>
  
  # float | null: Maximum distance (pixels) an instance can move between frames (optional)
  max_center_dist: <float | null>

  # float: Multiplier for angle difference penalty (default: 1.0)
  angle_diff_penalty_multiplier: <float>

  # list[str] | null: List of node names defining the front of the object for orientation (optional)
  front_nodes: <list[str] | null>
  
  # list[str] | null: List of node names defining the back of the object for orientation (optional)
  back_nodes: <list[str] | null>
  
  # float: Threshold below which frames are flagged as potential errors (default: 0)
  confidence_threshold: <float>
  
  # float: Multiplier for distance penalty (default: 2.0)
  distance_penalty_multiplier: <float>
  
  # int: Maximum number of tracks that can be created
  #      Set to number of objects you expect to track
  max_tracks: <int>
  
  # bool: Whether to use multiplicative threshold weighting (default: true)
  mult_thresh: <bool>
  
  # float: Trajectory overlap threshold for assignment (default: 0.01)
  overlap_thresh: <float>

trainer:
  # str: Device to use: "gpu", "cuda", or "cpu"
  accelerator: <str>
  
  # list[int]: List of device indices (e.g., [0] for first GPU)
  devices: <list[int]>
```

## CLI Usage

Most parameters can be set via CLI flags. For example:

```bash
dreem track ./data/inference \
  --checkpoint ./models/model.ckpt \
  --output ./results \
  --crop-size 84 \
  --max-tracks 2 \
  --max-dist 30 \
  --confidence-threshold 0.9 \
  --iou-mode mult
```

See `dreem track --help` or `dreem eval --help` for all available CLI options.

