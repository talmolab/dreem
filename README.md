# DREEM Relates Every Entity's Motion

[![CI](https://github.com/talmolab/dreem/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/dreem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/dreem/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/dreem)
[![Documentation](https://img.shields.io/badge/Documentation-dreem.sleap.ai-lightgrey)](https://dreem.sleap.ai)
[![code](https://img.shields.io/github/stars/talmolab/dreem)](https://github.com/talmolab/dreem)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/dreem?label=Latest)](https://github.com/talmolab/dreem/releases/)
[![PyPI](https://img.shields.io/pypi/v/dreem-track?label=PyPI)](https://pypi.org/project/dreem-track)

**DREEM** is an open-source framework for multiple object tracking in biological data. Train your own models, run inference on new data, and evaluate your results. DREEM supports a variety of detection types, including keypoints, bounding boxes, and segmentation masks.

<!-- TODO: Add GIF showing DREEM in action -->
<!-- ![DREEM Demo](docs/assets/dreem-demo.gif) -->

## Features

- ✅ **Command-Line & API Access:** Use DREEM via a simple CLI or integrate into your own Python scripts.
- ✅ **Pretrained Models:** Get started quickly with models trained specially for microscopy and animal domains.
- ✅ **Configurable Workflows:** Easily customize training and inference using YAML configuration files.
- ✅ **Visualization:** Visualize tracking results in your browser without any data leaving your machine, or use the SLEAP GUI for a more detailed view.
- ✅ **Examples:** Step-by-step notebooks and guides for common workflows.

<!-- TODO: Add GIF showing CLI usage -->
<!-- ![CLI Demo](docs/assets/cli-demo.gif) -->

## Installation

DREEM works best with Python 3.12. We recommend using [uv](https://docs.astral.sh/uv/) for package management.

In a new directory:
```bash
   uv venv && source .venv/bin/activate
   uv pip install dreem-track
   ```
or as a system-wide package that does not require a virtual environment:
```bash
   uv tool install dreem-track
   ```
Now dreem commands will be available without activating a virtual environment.

For more installation options and details, see the [Installation Guide](https://dreem.sleap.ai/installation/).

## Quickstart
### 1. Download Sample Data and Model

```bash
# Install huggingface-hub if needed
uv pip install huggingface_hub

# Download sample data
hf download talmolab/sample-flies --repo-type dataset --local-dir ./data

# Download pretrained model
hf download talmolab/animals-pretrained \
    --repo-type model \
    --local-dir ./models \
    --include "animals-pretrained.ckpt"
```

### 2. Run Tracking

```bash
dreem track ./data/inference \
    --checkpoint ./models/animals-pretrained.ckpt \
    --output ./results \
    --crop-size 70
```

### 3. Visualize Results

Results are saved as `.slp` files that can be opened directly in [SLEAP](https://sleap.ai) for visualization.

<!-- TODO: Add GIF showing visualization in SLEAP -->
<!-- ![SLEAP Visualization](docs/assets/sleap-visualization.gif) -->

For a more detailed walkthrough, check out the [Quickstart Guide](https://dreem.sleap.ai/quickstart/) or try the [Colab notebook](https://colab.research.google.com/github/talmolab/dreem/blob/docs/examples/quickstart.ipynb).

## Usage

### Training a Model

Train your own model on custom data:

```bash
dreem train ./data/train \
    --val-dir ./data/val \
    --crop-size 70 \
    --epochs 10
```

### Running Inference

Run tracking on new data with a pretrained model:

```bash
dreem track ./data/inference \
    --checkpoint ./models/my_model.ckpt \
    --output ./results \
    --crop-size 70
```

### Evaluating Results

Evaluate tracking accuracy against ground truth:

```bash
dreem eval ./data/test \
    --checkpoint ./models/my_model.ckpt \
    --output ./results \
    --crop-size 70
```

### Association Score Aggregation

DREEM pools per-instance association scores into a per-track score before matching. By default this
uses a **one-to-one, recency-weighted mean** over the lookback window
(`tracker.aggregation: "recency_mean"`).

Earlier versions took a softmax *within each reference frame* and then summed over each track's
instances, which made a track's score depend on how much of the window it occupied -- an instance
that disappeared for a few frames was scored against a skewed lookback, and the per-frame softmax
entangled tracks through frame co-occupancy. On held-out cross-video animal data, switching to the
one-to-one recency-weighted mean cut ID switches by 40-79% and improved HOTA/IDF1/AssA on most test
sets.

To restore the previous behavior, or to down-weight stale frames, set these in your inference config:

```yaml
tracker:
  aggregation: "softmax_sum"  # legacy per-reference-frame softmax + sum
  recency_tau: 5              # frames; null (default) weights the whole window equally
```

When animals leave view for a while, the track of an animal that vanished a few frames ago can
outscore that animal's own current track, because nothing discounts a track for how long it has been
unseen. `gap_penalty` subtracts `gap_penalty * log(frames since the track was last seen)` from each
track's score (tracks seen in the previous frame are unchanged; `0`, the default, turns it off). Pair
it with a small `recency_tau` so each track is scored on its newest instance. The best value depends
on the data; tune it on held-out clips:

```yaml
tracker:
  recency_tau: 0.1
  gap_penalty: 0.5
```

The IoU and `max_center_dist` terms compare each detection with the newest box of each track. When
animals move fast and steadily and pass close to each other (schooling fish, for example), the
detection nearest a track's newest box is often the other animal. `cv_reference: true` measures both
terms from where each track would be now at constant velocity: its newest box moved by the velocity
between its two newest instances in the window. Tracks with a single instance keep their newest box.
It helps when velocity predicts the next position better than the last position does. On animals that
stop and turn often (gerbils, mice in a home cage) it made tracking worse, so check it on held-out
clips:

```yaml
tracker:
  iou: "mult"
  cv_reference: true
```

See the [Inference Configuration Reference](https://dreem.sleap.ai/configs/inference/) for details.

For detailed usage instructions, see the [Usage Guide](https://dreem.sleap.ai/usage/).


## Documentation

- **[Installation Guide](https://dreem.sleap.ai/installation/)** - Detailed installation instructions
- **[Quickstart Guide](https://dreem.sleap.ai/quickstart/)** - Get started in minutes
- **[Usage Guide](https://dreem.sleap.ai/usage/)** - Complete workflow documentation
- **[Configuration Reference](https://dreem.sleap.ai/configs/)** - Customize training and inference
- **[API Reference](https://dreem.sleap.ai/reference/dreem/)** - Python API documentation
- **[Examples](https://dreem.sleap.ai/Examples/)** - Step-by-step notebooks

## Examples

We provide several example notebooks to help you get started:

- **[Quickstart Notebook](examples/quickstart.ipynb)** - Fly tracking demo with pretrained model
- **[End-to-End Demo](examples/dreem-demo.ipynb)** - Train, run inference, and evaluate
- **[Microscopy Demo](examples/microscopy-demo-simple.ipynb)** - Track cells in microscopy data

All notebooks are available on [Google Colab](https://colab.research.google.com/github/talmolab/dreem/tree/docs/examples).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and conventions
- Submitting pull requests
- Reporting issues

<!-- TODO: Add GIF showing contribution workflow -->
<!-- ![Contributing](docs/assets/contributing.gif) -->

## Citation

If you use DREEM in your research, please cite our paper:

```bibtex
@article{dreem2024,
  title={DREEM: Global Tracking Transformers for Biological Multi-Object Tracking},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

**Questions?** Open an issue on [GitHub](https://github.com/talmolab/dreem/issues) or visit our [documentation](https://dreem.sleap.ai).
