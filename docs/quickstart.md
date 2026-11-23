# Quickstart

<link rel="stylesheet" href="../css/quickstart.css">

Track animals or cells in your own videos in six steps. Steps 1 and 2 try DREEM on sample data; steps 3 to 6 are what you do with your own recordings.

<ol class="qs-steps">
  <li><a href="#1-install-dreem">Install</a></li>
  <li><a href="#2-try-the-sample-data">Try the sample</a></li>
  <li><a href="#3-prepare-your-data">Prepare your data</a></li>
  <li><a href="#4-choose-parameters">Choose parameters</a></li>
  <li><a href="#5-track-your-videos">Track your videos</a></li>
  <li><a href="#6-check-the-results">Check the results</a></li>
</ol>

!!! info "What you need"
    - Python 3.12 and [uv](https://docs.astral.sh/uv/) (or pip).
    - For your own data: videos, plus detections from a pose or segmentation tool such as SLEAP or CellPose (step 3).
    - A GPU is optional: steps 1 and 2 run on a CPU.

Prefer a notebook? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talmolab/dreem/blob/main/examples/quickstart.ipynb) runs steps 1 and 2 in your browser.

## 1. Install DREEM

<p class="qs-time">About a minute</p>

In a new folder:

```bash
uv venv && source .venv/bin/activate
uv pip install dreem-track
```

Other ways to install (pip, `uv tool`) and GPU support: [Installation](installation.md).

!!! success "You should see"
    `dreem --help` lists the commands `track`, `eval`, `convert`, `train`, `render` and `system`.

## 2. Try the sample data

<p class="qs-time">About 15 minutes on a CPU, faster on a GPU</p>

Download two short videos of two flies, with their detections, from Hugging Face (260 MB; the `hf` command comes with DREEM):

```bash
hf download talmolab/sample-flies --repo-type dataset --local-dir ./data
```

Track the video in `data/inference` with the pretrained animal model:

```bash
dreem track ./data/inference --checkpoint animals --output ./results --crop-size 70 --max-tracks 2
```

- `--checkpoint animals` downloads the pretrained animal model on first use. `microscopy` is the model for cells; see [Pretrained Models](models_pg.md).
- `--crop-size 70` is the side, in pixels, of the square DREEM crops around each fly.
- `--max-tracks 2` says there are two flies.

!!! success "You should see"
    A progress bar, then the saved file and a summary:

    ```text
    Predicting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80/80 0:13:00 • 0:00:00 0.75it/s
    Saved: .../results/190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.<date>-<time>.slp
      Frames: 2560, Tracks: 2, IDs: [0, 1]
    ```

    Two tracks, one per fly, over all 2,560 frames. Look at them in step 6.

??? failure "If it fails"
    - **`hf: command not found`**: the environment from step 1 is not active; run `source .venv/bin/activate`.
    - **`dreem track` cannot download the model**: the first run needs internet access to Hugging Face. Offline, pass a downloaded checkpoint file instead: `--checkpoint ./animals-pretrained.ckpt`.

## 3. Prepare your data

<p class="qs-time">Depends on your detection tool</p>

DREEM links detections across frames into tracks; it does not find the animals or cells itself. For each recording you need:

- [ ] **A video**: `.mp4` or `.avi`, or a `.tif` stack for microscopy.
- [ ] **Detections for its frames**: poses or centroids in a SLEAP `.slp` file (from [SLEAP](https://sleap.ai), or from TrackMate through [`dreem convert`](cli.md#convert)), or Cell Tracking Challenge masks (from [CellPose](https://www.cellpose.org) or [Ilastik](https://www.ilastik.org)). Identities are not needed: assigning them is DREEM's job.
- [ ] **Matching names in one folder**: `x.mp4` pairs with `x.slp`, `x.predictions.slp` or `x.mp4.predictions.slp`.

```text
my_data/
├── mouse_day1.mp4
├── mouse_day1.slp
├── mouse_day2.mp4
└── mouse_day2.predictions.slp
```

The sample's `data/inference/` folder has this layout: one video and its `.slp`. For Cell Tracking Challenge folders, see the [Usage guide](usage.md#directory-structure).

## 4. Choose parameters

<p class="qs-time">A few minutes</p>

Two settings depend on your data:

| Setting | What it is | How to choose |
|---|---|---|
| `--crop-size` (required) | The side, in pixels, of the square DREEM crops around each animal or cell and looks at. | Large enough to hold the whole animal or cell. |
| `--max-tracks` | How many tracks DREEM may create. | The number of animals, when it never changes. Leave it out for cells that enter, leave or divide. |

**[Choosing Parameters](parameters.md)** reads your `.slp` (and video) in your browser, measures your animals, and suggests these and the other settings with a command you can copy.

!!! note "The sample's crop size"
    The sample flies are about 102 px long, so Choosing Parameters suggests a whole-fly crop of 104 for them; this Quickstart uses 70. The pretrained animal model was trained on crops of 64 to 128 px.

## 5. Track your videos

<p class="qs-time">Depends on video length and hardware</p>

Run `dreem track` on your folder with your values from step 4:

=== "Animals"

    ```bash
    dreem track ./my_data --checkpoint animals --output ./results --crop-size 104 --max-tracks 2
    ```

=== "Cells"

    ```bash
    dreem track ./my_data --checkpoint microscopy --output ./results --crop-size 32
    ```

Add `--output-format both` to also get a CSV of trajectories. Every option is in the [CLI reference](cli.md).

!!! success "You should see"
    One `<video>.dreem_inference.<date>-<time>.slp` per video in `./results` (`.tif` masks for Cell Tracking Challenge data), with one track per animal or cell.

## 6. Check the results

<p class="qs-time">A few minutes</p>

=== "In your browser"

    Open the [Live Visualizer](visualizer.md) and choose a result `.slp` and its video. Each track is drawn in its own color, and nothing is uploaded.

=== "In SLEAP"

    Open a result in the SLEAP GUI ([install SLEAP](https://docs.sleap.ai/latest/)):

    ```bash
    sleap-label results/<your_output_file>.slp
    ```

    The GUI needs a display, so it may not open on a remote server.

=== "Against ground truth"

    If you have proofread tracks for some videos, `dreem eval` scores DREEM on them (MOTA, IDF1, identity switches). On the sample:

    ```bash
    dreem eval ./data/test --checkpoint animals --output ./eval-results --crop-size 70 --max-tracks 2
    ```

If identities swap when animals move fast or come close, see the `--max-dist` advice in [Choosing Parameters](parameters.md).

## Next steps

- [Usage guide](usage.md): config files, training and every workflow.
- [End-to-end demo](Examples/dreem-demo.md): train and evaluate your own model.
- [Microscopy demo](Examples/microscopy-demo.md): detect cells with CellPose and track them with the pretrained microscopy model.
- [CLI reference](cli.md): every command and option.
