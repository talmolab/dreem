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

Load the original images and the tracked output:

```python
images = tifffile.TiffSequence(os.path.join(data_path, "*.tif")).asarray().astype(np.uint16)
tracked = tifffile.imread(tracked_path).astype(np.uint16)
```

Then use an interactive slider to browse frames with tracked identity overlays:

```python
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, interact


def browse(z=0):
    """Browse frames with tracked identity overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(images[z], cmap="gray")
    masked = np.ma.masked_where(tracked[z] == 0, tracked[z])
    ax.imshow(masked, cmap="tab20", alpha=0.6, interpolation="nearest")
    ax.set_title(f"Frame {z}")
    plt.show()

interact(browse, z=IntSlider(min=0, max=len(images)-1, step=1))
```
