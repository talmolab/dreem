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

### Option 1: Upload Your Own Data

Upload your files directly using the **Colab file browser**: click the folder icon in the left sidebar, navigate into `./data/`, and drag and drop your files in.

Supported input formats:

- **TIFF directory**: A folder of individual TIFF frame files. Upload into `./data/<your_folder_name>/<video_name>/`.
- **TIFF stack**: A single multi-page TIFF file. Upload to `./data/` and set `data_path` to the file path.
- **Video** (`.avi`, `.mp4`): Upload to `./data/`, then run the video conversion cell below.

> If you do not have your own data, skip ahead to **Option 2** to download our sample dataset.


#### Convert a video or TIFF stack to individual TIFF frames

Skip this cell if you already have a directory of individual TIFF frames. If you uploaded a `.avi`, `.mp4`, or single multi-page `.tif` file, set `input_path` below and run the cell.

```python
input_path = "./data/your_video.mp4"  # <-- update this to your uploaded file

base = os.path.splitext(os.path.basename(input_path))[0]
data_path = os.path.abspath(f"./data/{base}/{base}")
os.makedirs(data_path, exist_ok=True)

# Detect file type and convert to individual TIFF frames
ext = os.path.splitext(input_path)[1].lower()

if ext in (".tif", ".tiff"):
    # Multi-page TIFF stack
    stack = tifffile.imread(input_path)
    if stack.ndim == 2:
        raise ValueError("Input TIFF is a single frame, not a stack.")
    for i in range(stack.shape[0]):
        tifffile.imwrite(os.path.join(data_path, f"frame_{i:05d}.tif"), stack[i])
    print(f"Extracted {stack.shape[0]} frames from TIFF stack to: {data_path}")
elif ext in (".avi", ".mp4"):
    # Video file
    video = sio.load_video(input_path)
    for i, frame in enumerate(video):
        frame = frame[..., 0] if frame.ndim == 3 else frame
        tifffile.imwrite(os.path.join(data_path, f"frame_{i:05d}.tif"), frame)
    print(f"Extracted {len(video)} frames from video to: {data_path}")
else:
    raise ValueError(f"Unsupported file type: {ext}. Use .tif, .tiff, .avi, or .mp4")
```

### Option 2: Use Sample Data

If you don't have your own data, run the cell below to download our sample microscopy dataset from HuggingFace. The download includes:

- **DynamicNuclearNet** — cell nuclei imaged with fluorescence microscopy. A single tiff stack of 42 frames. Data credit to Van Valen Lab (https://doi.org/10.1101/803205)

```python
!hf download talmolab/microscopy-demo --repo-type dataset --local-dir ./data
```

## Detection

Here we use CellPose to create segmentation masks for our instances.

Update `data_path` below to point to the directory containing your individual TIFF frame files. If you are using the sample data from Option 2, the default path is already correct.

```python
data_path = os.path.abspath("./data/dynamicnuclearnet/test_1")  # <-- update if using your own data

# Derive paths from data_path
dataset_dir = os.path.dirname(data_path)
segmented_path = os.path.join(dataset_dir, os.path.basename(data_path) + "_GT", "TRA")
results_path = os.path.abspath("./results")

print(f"Data path:      {data_path}")
print(f"Segmented path: {segmented_path}")
print(f"Results path:   {results_path}")
```

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
!dreem track {dataset_dir} --checkpoint ./models/pretrained-microscopy.ckpt --output {results_path} --video-type tif --crop-size {instance_diameter_px}

print(f"\nResults saved to: {results_path}")
```

### Visualize the results

Load the original images and the tracked segmentation masks from the output directory:

```python
images = tifffile.TiffSequence(os.path.join(data_path, "*.tif")).asarray().astype(np.uint16)
labels = tifffile.TiffSequence(os.path.join(segmented_path, "*.tif")).asarray().astype(np.uint16)
```

Then use an interactive slider to browse frames with track overlays:

```python
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, interact


def browse(z=0):
    """Browse frames with segmentation overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(images[z], cmap="gray")
    masked = np.ma.masked_where(labels[z] == 0, labels[z])
    ax.imshow(masked, cmap="tab20", alpha=0.6, interpolation="nearest")
    ax.set_title(f"Z={z}")
    plt.show()

interact(browse, z=IntSlider(min=0, max=len(images)-1, step=1))
```
