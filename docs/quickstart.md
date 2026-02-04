# Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/docs/examples/quickstart.ipynb)

DREEM operations can be performed either through the command-line interface or through the API (for more advanced users). The CLI provides commands for training, tracking, evaluation, and visualization. All operations can be customized via config files. For a more in-depth walkthrough, see the [Examples](Examples/dreem-demo.md) section. For a complete reference of all commands and options, see [the API Reference](https://dreem.sleap.ai/reference/dreem/).

This quickstart walks you through tracking a social interaction between two flies from the SLEAP [fly32 dataset](https://sleap.ai/datasets.html#fly32) using a pretrained model. **Runtime**: ~5–10 minutes. **Hardware**: CPU is sufficient. See [Installation](installation.md) if you haven't installed DREEM yet.

---

## Step 1: Install dependencies (Colab only)

If you're running in Colab, install DREEM and helpers in a cell:

```python
%pip install uv
!uv pip install dreem-track huggingface_hub opencv-python-headless
!apt-get install -y ffmpeg 2>/dev/null || echo "ffmpeg already installed or not on Linux"
```

If you're running locally, use the [installation guide](installation.md).

---

## Step 2: Download sample data

Download the sample fly dataset (videos, `.slp` detections, and configs):

```bash
hf download talmolab/sample-flies --repo-type dataset --local-dir ./data
```

Ensure `huggingface_hub` is installed (`pip install huggingface_hub`); the `hf` CLI comes with it.

---

## Step 3: Download pretrained model

Download a pretrained DREEM model (trained on mice, flies, zebrafish):

```bash
hf download talmolab/animals-pretrained animals-pretrained.ckpt --local-dir=./models
```

---

## Step 4: Run tracking

Run tracking on the inference data. Use a crop size that matches your instance size (here, 70 pixels) and set `--max-tracks` to the number of animals (2 for this dataset):

```bash
dreem track ./data/inference --checkpoint ./models/animals-pretrained.ckpt --output ./results --crop-size 70 --max-tracks 2
```

---

## Step 5: Evaluate tracking (optional)

If you have ground truth labels (e.g. in `./data/test`), run evaluation to get metrics (MOTA, IDF1, ID switches):

```bash
dreem eval ./data/test --checkpoint ./models/animals-pretrained.ckpt --output "./eval-results" --crop-size 70 --max-tracks 2
```

---

## Step 6: Visualize results

**Option A – SLEAP GUI (full pose keypoints)**  
Install SLEAP ([https://docs.sleap.ai/latest/](https://docs.sleap.ai/latest/)) and open the output `.slp` in SLEAP:

```bash
sleap-label results/<your_output_file>.slp
```

The SLEAP GUI may not render on a remote server.

**Option B – Quick animation in Python**  
From the `examples/` directory you can use the helper script and load the latest result:

```python
import os
import sleap_io as sio
import pandas as pd
import numpy as np
from create_animation import create_tracking_animation

results_dir = "./results"
result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.slp') and 'dreem_inference' in f])
result_path = os.path.join(results_dir, result_files[-1])
pred_slp = sio.load_slp(result_path)

list_frames = []
for lf in pred_slp:
    for instance in lf.instances:
        centroid = np.nanmean(instance.numpy(), axis=0)
        track_id = int(instance.track.name) if instance.track else -1
        list_frames.append({"frame_id": lf.frame_idx, "track_id": track_id, "centroid": centroid})
df = pd.DataFrame(list_frames)

video_path = "./data/inference/190719_090330_wt_18159206_rig1.2@15000-17560.mp4"  # or glob for *.mp4
create_tracking_animation(video_path, df, fps=15, max_frames=100)
```

---

## Next steps

- Run the [end-to-end demo](Examples/dreem-demo.md) to train a model, evaluate, and visualize in a single notebook.
- See the [Usage guide](usage.md) for CLI options, config files, and workflows.
- Use SLEAP to generate detections on your own videos, then run DREEM tracking with the same pretrained checkpoint.
