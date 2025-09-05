# /// script
# dependencies = [
#     "numpy",
#     "tifffile",
#     "cellpose",
# ]
# ///

import os
import numpy as np
import tifffile
from cellpose import models

data_path = "./data/dynamicnuclearnet/test_1"
segmented_path = "./data/dynamicnuclearnet/test_1_GT/TRA"
os.makedirs(segmented_path, exist_ok=True)

diam_px = 25

tiff_files = [
    f for f in os.listdir(data_path) if f.endswith(".tif") or f.endswith(".tiff")
]
stack = np.stack([tifffile.imread(os.path.join(data_path, f)) for f in tiff_files])
frames, Y, X = stack.shape

channels = [0, 0]
# use builtin latest model
model = models.CellposeModel(gpu=True)
all_masks = np.zeros_like(stack)
for i, img in enumerate(stack):
    masks, flows, styles = model.eval(
        img,
        diameter=diam_px,
        cellprob_threshold=0.0,
        channels=channels,
        z_axis=None,
    )
    all_masks[i] = masks

os.makedirs(segmented_path, exist_ok=True)
for i, (mask, filename) in enumerate(zip(all_masks, tiff_files)):
    new_tiff_path = os.path.join(segmented_path, f"{os.path.splitext(filename)[0]}.tif")
    print(f"exporting frame {i} to tiff at {new_tiff_path}")
    tifffile.imwrite(new_tiff_path, mask)
