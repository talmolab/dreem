[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/main/examples/microscopy-demo.ipynb)

## DREEM workflow for microscopy
### From raw tiff stacks to tracked identities

This notebook will walk you through the typical workflow for microscopy identity tracking. We start with a raw tiff stack, pass it through an off-the-shelf detection model, and feed those detections into DREEM.

To run this demo, you can use your own data or our sample data.

> **GPU recommended.** CellPose segmentation and DREEM tracking both benefit from GPU acceleration. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU** before running.

### Install DREEM

```python
!uv pip install dreem-track cellpose tifffile
```

### Import necessary packages

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio
import tifffile
import torch
from huggingface_hub import hf_hub_download

from dreem.utils import run_cellpose_segmentation
```
### Download a pretrained model

```python
model_save_dir = "./models"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs("./data", exist_ok=True)
model_path = hf_hub_download(
    repo_id="talmolab/microscopy-pretrained",
    filename="pretrained-microscopy.ckpt",
    local_dir=model_save_dir,
)
```

## Data

Set `data_path` below to point to your data. Supported formats:

- **Directory of TIFFs** (default): Set `data_path` to the folder containing individual `.tif` frame files.
- **TIFF stack**: Set `data_path` to a single multi-page `.tif` file — it will be split into individual frames automatically.
- **Video**: Set `data_path` to a `.mp4` or `.avi` file — frames will be extracted automatically.

Leave `data_path = None` to download and use our sample dataset (**DynamicNuclearNet** — 42 frames of fluorescent cell nuclei, credit: [Van Valen Lab](https://doi.org/10.1101/803205)).

```python
# data_path = "./data/my_folder/my_tiffs"       # directory of individual .tif frames
# data_path = "./data/my_stack.tif"              # multi-page TIFF stack
# data_path = "./data/my_video.mp4"              # video file (.avi, .mp4)
data_path = None                                  # None = download sample dataset
```

```python
if data_path is None:
    # Download sample dataset
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="talmolab/microscopy-demo",
        repo_type="dataset",
        local_dir="./data",
    )
    data_path = os.path.abspath("./data/dynamicnuclearnet/test_1")
    print(f"Downloaded sample data to: {data_path}")
else:
    data_path = os.path.abspath(data_path)

# If data_path is a file (video or TIFF stack), convert to a directory of TIFFs
if os.path.isfile(data_path):
    ext = os.path.splitext(data_path)[1].lower()
    base = os.path.splitext(os.path.basename(data_path))[0]
    tiff_dir = os.path.join(os.path.dirname(data_path), base, base)
    os.makedirs(tiff_dir, exist_ok=True)

    if ext in (".tif", ".tiff"):
        stack = tifffile.imread(data_path)
        if stack.ndim == 2:
            raise ValueError("Input TIFF is a single frame, not a stack.")
        for i in range(stack.shape[0]):
            tifffile.imwrite(os.path.join(tiff_dir, f"frame_{i:05d}.tif"), stack[i])
        print(f"Extracted {stack.shape[0]} frames from TIFF stack to: {tiff_dir}")
    elif ext in (".avi", ".mp4"):
        video = sio.load_video(data_path)
        for i, frame in enumerate(video):
            frame = frame[..., 0] if frame.ndim == 3 else frame
            tifffile.imwrite(os.path.join(tiff_dir, f"frame_{i:05d}.tif"), frame)
        print(f"Extracted {len(video)} frames from video to: {tiff_dir}")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .tif, .tiff, .avi, or .mp4")
    data_path = tiff_dir

# Derive all paths from data_path
dataset_dir = os.path.dirname(data_path)
segmented_path = os.path.join(dataset_dir, os.path.basename(data_path) + "_GT", "TRA")
results_path = os.path.abspath("./results")

print(f"Data path:      {data_path}")
print(f"Segmented path: {segmented_path}")
print(f"Results path:   {results_path}")
```

## Detection

Here we use CellPose to create segmentation masks for our instances.

Set the approximate diameter (in pixels) of the instances you want to segment:

```python
instance_diameter_px = 25
probability_threshold = 0.0
```

### Run detection model

```python
use_gpu = torch.cuda.is_available()

masks = run_cellpose_segmentation(
    data_path,
    segmented_path,
    diameter=instance_diameter_px,
    gpu=use_gpu,
    cellprob_threshold=probability_threshold,
)

# Load the original stack and masks for visualization
tiff_files = sorted(
    f for f in os.listdir(data_path) if f.endswith(".tif") or f.endswith(".tiff")
)
first_img = tifffile.imread(os.path.join(data_path, tiff_files[0]))
first_mask = tifffile.imread(
    os.path.join(segmented_path, f"{os.path.splitext(tiff_files[0])[0]}.tif")
)
```

#### View the segmentation result and original image

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(first_mask)
ax1.set_title("Segmentation Mask")
ax2.imshow(first_img)
ax2.set_title("Original Image")
plt.tight_layout()
plt.show()
```

## Tracking

This assumes you have run the CellPose segmentation step above. The output is a single TIFF file with all frames, as well as the configuration used for tracking (for reproducibility).

```python
gpu_flag = "--gpu" if torch.cuda.is_available() else "--no-gpu"

!dreem track {dataset_dir} --checkpoint ./models/pretrained-microscopy.ckpt --output {results_path} --video-type tif --crop-size {instance_diameter_px} {gpu_flag}

# Find the tracking output (most recent .dreem_inference.tif in results)
import glob

tracked_files = sorted(glob.glob(os.path.join(results_path, "*.dreem_inference.*.tif")))
tracked_path = tracked_files[-1]
print(f"\nResults saved to: {tracked_path}")
```

### Visualize the results

Load the raw images, detection masks, and tracked output, then browse interactively.

The viewer shows three panels:
- **Raw image** — original microscopy frame
- **Detections** — CellPose segmentation masks (untracked)
- **Tracked** — DREEM identity assignments with ID labels and trajectory trails

```python
from collections import defaultdict

# Load raw images, detection masks, and tracked output
images = tifffile.TiffSequence(os.path.join(data_path, "*.tif")).asarray().astype(np.uint16)
tracked = tifffile.imread(tracked_path).astype(np.uint16)

seg_files = sorted(
    f for f in os.listdir(segmented_path) if f.endswith(".tif") or f.endswith(".tiff")
)
detections = np.stack([tifffile.imread(os.path.join(segmented_path, f)) for f in seg_files])

# Build a stable color map: one consistent color per track ID across all frames
track_ids = sorted(set(np.unique(tracked)) - {0})
rng = np.random.RandomState(42)
base_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
if len(track_ids) > 20:
    extra = plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]
    base_colors = np.vstack([base_colors, extra])
shuffled_idx = rng.permutation(len(base_colors))
id_to_color = {}
for i, tid in enumerate(track_ids):
    id_to_color[tid] = base_colors[shuffled_idx[i % len(base_colors)]]

# Precompute centroids per frame for trajectory trails
centroids = defaultdict(dict)  # centroids[frame_idx][track_id] = (row, col)
for t in range(tracked.shape[0]):
    ids_in_frame = set(np.unique(tracked[t])) - {0}
    if ids_in_frame:
        labels = tracked[t]
        for tid in ids_in_frame:
            ys, xs = np.where(labels == tid)
            centroids[t][tid] = (ys.mean(), xs.mean())

print(f"Frames: {len(images)}")
print(f"Unique tracks: {len(track_ids)}")
print(f"Track IDs: {track_ids}")
```

Then use an interactive slider to browse frames with the three-panel view:

```python
from ipywidgets import IntSlider, interact

TRAIL_LENGTH = 10  # number of past frames to show trajectory trails


def colorize_masks(mask_frame, color_map):
    """Convert a label mask to an RGBA image using a color map."""
    h, w = mask_frame.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for tid in set(np.unique(mask_frame)) - {0}:
        color = color_map.get(tid, [0.5, 0.5, 0.5])
        region = mask_frame == tid
        rgba[region, :3] = color
        rgba[region, 3] = 0.6
    return rgba


def browse(z=0):
    """Browse frames: raw image, detection masks, and tracked identities."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Raw image
    ax1.imshow(images[z], cmap="gray")
    ax1.set_title("Raw Image")
    ax1.axis("off")

    # Panel 2: Detection masks (untracked)
    ax2.imshow(images[z], cmap="gray")
    det_mask = detections[z]
    det_rgba = np.zeros((*det_mask.shape, 4), dtype=np.float32)
    det_ids = set(np.unique(det_mask)) - {0}
    det_colors = plt.cm.Set3(np.linspace(0, 1, max(len(det_ids), 1)))
    for i, did in enumerate(sorted(det_ids)):
        region = det_mask == did
        det_rgba[region, :3] = det_colors[i % len(det_colors)][:3]
        det_rgba[region, 3] = 0.5
    ax2.imshow(det_rgba, interpolation="nearest")
    ax2.set_title(f"Detections ({len(det_ids)} cells)")
    ax2.axis("off")

    # Panel 3: Tracked identities with labels and trails
    ax3.imshow(images[z], cmap="gray")
    tracked_rgba = colorize_masks(tracked[z], id_to_color)
    ax3.imshow(tracked_rgba, interpolation="nearest")

    # Draw centroid labels and trajectory trails
    for tid, (cy, cx) in centroids[z].items():
        color = id_to_color.get(tid, [0.5, 0.5, 0.5])

        # Trail: connect centroids from recent frames
        trail_x, trail_y = [], []
        for prev in range(max(0, z - TRAIL_LENGTH), z + 1):
            if tid in centroids[prev]:
                py, px = centroids[prev][tid]
                trail_x.append(px)
                trail_y.append(py)
        if len(trail_x) > 1:
            ax3.plot(trail_x, trail_y, color=color, linewidth=1.5, alpha=0.8)

        # ID label at centroid
        ax3.text(
            cx, cy, str(tid),
            color="white", fontsize=7, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.7, edgecolor="none"),
        )

    ids_in_frame = set(np.unique(tracked[z])) - {0}
    ax3.set_title(f"Tracked ({len(ids_in_frame)} cells, {len(track_ids)} total tracks)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


interact(browse, z=IntSlider(min=0, max=len(images) - 1, step=1, description="Frame:"))
```
