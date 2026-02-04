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

- **Command-Line & API Access:** Use DREEM via a simple CLI or integrate into your own Python scripts
- **Configurable Workflows:** Easily customize training and inference using CLI arguments or config files
- **Pretrained Models:** Get started quickly with models trained for [microscopy](https://huggingface.co/talmolab/microscopy-pretrained) and [animal](https://huggingface.co/talmolab/animals-pretrained) domains. Available on Hugging Face.
- **Visualization:** Tracking outputs are directly compatible with SLEAP's GUI
- **Examples:** Step-by-step notebooks and guides for common workflows

<!-- TODO: Add GIF showing CLI usage -->
<!-- ![CLI Demo](docs/assets/cli-demo.gif) -->

## Installation

DREEM works best with Python 3.12. We recommend using [uv](https://docs.astral.sh/uv/) for package management.

In a new directory:
```bash
   uv init
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
