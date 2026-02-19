[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/main/examples/microscopy-demo.ipynb)

## DREEM workflow for microscopy
### From raw tiff stacks to tracked identities

This notebook will walk you through the typical workflow for microscopy identity tracking. We start with a raw tiff stack, pass it through an off-the-shelf detection model, and feed those detections into DREEM. 

This notebook uses a simple entrypoint into the tracking code. You only need to specify a configuration file, and a few lines of code!

To run this demo, we have provided sample data and model checkpoints. A GPU is recommended if you run the CellPose segmentation step, otherwise the tracking will run on a CPU.

### Install DREEM

```python
!uv pip install dreem-track cellpose tifffile
```

### Import necessary packages

```python
import os
import torch
import numpy as np
import tifffile
import sleap_io as sio
import matplotlib.pyplot as plt
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

- **TIFF directory**: Upload the individual TIFF frame files into `./data/<your_folder_name>/<video_name>`. For example, `./data/organelles/lysosomes-1`.
- **Video** (`.avi`, `.mp4`): Upload the video file to `./data/`, then run the conversion cell below

> If you do not have your own data, skip ahead to **Option 2** to download our sample dataset.


#### Convert a video to TIFF frames. Skip this cell if you uploaded a TIFF directory.

If you uploaded a `.avi` or `.mp4` file, set `video_path` below and run the cell to convert it to individual TIFF frames.

```python
video_path = "./data/your_video.mp4"  # <-- update this to your uploaded file

base_name = os.path.splitext(os.path.basename(video_path))[0]
custom_data_path = f"./data/{base_name}/{base_name}"
custom_segmented_path = f"./data/{base_name}/{base_name}_GT/TRA"
os.makedirs(custom_data_path, exist_ok=True)
os.makedirs(custom_segmented_path, exist_ok=True)

video = sio.load_video(video_path)
for i, frame in enumerate(video):
    frame = frame[..., 0] if frame.ndim == 3 else frame
    with tifffile.TiffWriter(
        os.path.join(custom_data_path, f"frame_{i:05}.tif"), mode="w"
    ) as writer:
        writer.write(frame)

print(f"Done. TIFF stack saved to: {custom_data_path}")
```

### Option 2: Use Sample Data

If you don't have your own data, run the cell below to download our sample microscopy dataset from HuggingFace. The download includes:

- **DynamicNuclearNet** — cell nuclei imaged with fluorescence microscopy. A single tiff stack of 42 frames. Data credit to Van Valen Lab (https://doi.org/10.1101/803205)

```python
!hf download talmolab/microscopy-demo --repo-type dataset --local-dir ./data
```

## Detection

Here we use CellPose to create segmentation masks for our instances.

Update the path below to the path to the directory containing the tiff files. If you are using our sample data, the path is already set.

```python
data_path = "./data/dynamicnuclearnet/test_1" # <-- update this to the path to your data

segmented_path = f"{data_path}_GT/TRA"
os.makedirs(segmented_path, exist_ok=True)
base_name = os.path.dirname(data_path)
```

Set the approximate diameter (in pixels) of the instances you want to segment

```python
diam_px = 25
```

### Run detection model

```python
gpu_flag = "--gpu" if torch.cuda.is_available() else "--no-gpu"

# runs Cellpose and outputs files to segmented_path
masks = run_cellpose_segmentation(
    data_path,
    segmented_path,
    diameter=diam_px,
    gpu=gpu_flag,
)
# Load the original stack and masks for visualization
tiff_files = [
    f for f in os.listdir(data_path) if f.endswith(".tif") or f.endswith(".tiff")
]
tiff_files.sort()  # Ensure consistent ordering
first_img = tifffile.imread(os.path.join(data_path, tiff_files[0]))
mask_path = os.path.join(segmented_path, f"{os.path.splitext(tiff_files[0])[0]}.tif")
first_mask = tifffile.imread(mask_path)
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

This assumes you have the run the CellPose segmentation step. The output is a single tiff file with all frames, as well as configurations used for tracking (this will help reproduce results). The location is what you set below with the --output flag.

```
!dreem track {base_name} --checkpoint ./models/pretrained-microscopy.ckpt --output ./results-dnn --video-type tif --crop-size 32
```

### Visualize the results
To visualize the tracked tiff stacks, you can use tools like ImageJ, Fiji, or Napari plugins.
