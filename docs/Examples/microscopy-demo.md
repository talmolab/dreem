[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/main/examples/microscopy-demo.ipynb)

## DREEM workflow for microscopy
### From raw tiff stacks to tracked identities

This notebook will walk you through the typical workflow for microscopy identity tracking. We start with a raw tiff stack, pass it through an off-the-shelf detection model, and feed those detections into DREEM. 

This notebook uses a simple entrypoint into the tracking code. You only need to specify a configuration file, and a few lines of code!

To run this demo, we have provided sample data and model checkpoints. A GPU is recommended if you run the CellPose segmentation step, otherwise the tracking will run on a CPU.

#### Install DREEM

```python
!uv pip install dreem-track
!uv pip install cellpose tifffile
```

#### Import necessary packages

```python
import os
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from dreem.utils import run_cellpose_segmentation
import subprocess
```

#### Download a pretrained model, configs and some data

```python
model_save_dir = "./models"
config_save_dir = "./configs"
os.makedirs(config_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
```

```python
model_path = hf_hub_download(
    repo_id="talmolab/microscopy-pretrained",
    filename="pretrained-microscopy.ckpt",
    local_dir=model_save_dir,
)

config_path = hf_hub_download(
    repo_id="talmolab/microscopy-pretrained",
    filename="sample-microscopy-config.yaml",
    local_dir=config_save_dir,
)
```

```python
!hf download talmolab/microscopy-demo --repo-type dataset --local-dir ./data
```

## Detection

Here we use CellPose to create segmentation masks for our instances. **If you want to skip this stage**, we have provided segmentation masks for the lysosomes dataset located at ./data/lysosomes, and you can go straight ahead to the "Tracking" section below

#### Run CellPose segmentation with uv

```python
data_path = "./data/dynamicnuclearnet/test_1"
segmented_path = "./data/dynamicnuclearnet/test_1_GT/TRA"
os.makedirs(segmented_path, exist_ok=True)
```

Set the approximate diameter (in pixels) of the instances you want to segment

```python
diam_px = 25
```

#### Run detection model

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

### View the segmentation result and original image

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
Note that the segmented masks are saved to ./data/dynamicnuclearnet/test_1_GT/TRA; in general, any segmented masks are expected to be in a directory with the same name as the original data, with _GT/TRA appended to the end.

The command below assumes you have run the CellPose segmentation step, and that the segmented masks are saved to ./data/dynamicnuclearnet/test_1_GT/TRA. If you have not run the segmentation step, you can use the following command to track the lysosome data that we have provided: 

```
!dreem track ./data/lysosomes --checkpoint ./models/pretrained-microscopy.ckpt --output ./results-lyso --video-type tif --crop-size 22
```
If you did run the CellPose segmentation step, you can use the following command to track the data:
```python
!dreem track ./data/dynamicnuclearnet --checkpoint ./models/pretrained-microscopy.ckpt --output ./results-dnn --video-type tif --crop-size 32
```

### Visualize the results
To visualize the tracked tiff stacks, you can use tools like ImageJ, Fiji, or Napari plugins.
