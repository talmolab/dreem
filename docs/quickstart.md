# Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/main/examples/quickstart.ipynb)

This quickstart walks you through tracking a social interaction between two flies from the SLEAP [flies13 dataset](https://docs.sleap.ai/dev/reference/datasets/#flies13) using a pretrained model.

**Runtime**: ~5–10 minutes.  
**Hardware**: CPU is sufficient.  

---

## Step 1: Install dependencies

```python
!uv pip install dreem-track
```
See the [installation guide](installation.md) for more details.

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
hf download talmolab/dreem-animals-pretrained animals-pretrained.ckpt --local-dir=./models
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
dreem eval ./data/test --checkpoint ./models/animals-pretrained.ckpt --output ./eval-results --crop-size 70 --max-tracks 2
```

---

## Step 6: Visualize results

**Option A – DREEM Visualizer (browser-based)**  
Head to the [live visualizer](visualizer.md) to visualize your tracking results in your browser without any data leaving your machine.

**Option B – SLEAP GUI (full pose keypoints)**  
Install SLEAP ([https://docs.sleap.ai/latest/](https://docs.sleap.ai/latest/)) and open the output `.slp` in SLEAP:

```bash
sleap-label results/<your_output_file>.slp
```
The SLEAP GUI may not render on a remote server.

## Next steps

- Run the [end-to-end demo](Examples/dreem-demo.md) to train and evaluate a model.
- Run the [microscopy demo](Examples/microscopy-demo.md) to use CellPose for detection and a pretrained microscopy model for tracking.
- See the [Usage guide](usage.md) for CLI options, config files, and workflows.
- Use SLEAP to generate detections on your own videos, then run DREEM tracking with the same pretrained checkpoint.
- For a complete reference of all commands and options, see [the API Reference](https://dreem.sleap.ai/reference/dreem/).