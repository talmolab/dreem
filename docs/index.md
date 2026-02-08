# DREEM Relates Every Entity's Motion

[![CI](https://github.com/talmolab/dreem/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/dreem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/dreem/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/dreem)
[![code](https://img.shields.io/github/stars/talmolab/dreem)](https://github.com/talmolab/dreem)
[![Release](https://img.shields.io/github/v/release/talmolab/dreem?label=Latest)](https://github.com/talmolab/dreem/releases/)
[![PyPI](https://img.shields.io/pypi/v/dreem-track?label=PyPI)](https://pypi.org/project/dreem-track)

Welcome to the documentation for **DREEM** – an open-source tool for multiple object tracking. It enables you to track identities of objects in video data. You can run tracking using our pretrained models for animals and microscopy, or train a new model on your own data. In either case, DREEM takes in detection outputs in the form of pose keypoints or segmentation masks, and outputs identities of objects in the video. For an interactive demo, visit our [Hugging Face demo page](https://huggingface.co/spaces/talmolab/dreem).

<div class="termy">
```console
$ uv venv && source .venv/bin/activate
$ uv pip install dreem-track
$ dreem track ./data --checkpoint ./models/pretrained.ckpt --output ./results --crop-size 128
Running tracking...
---> 100%
Tracking complete! Results saved to results/
```

</div>

## Key Features

- ✅ **Command-Line & API Access:** Use DREEM via a simple CLI or integrate into your own Python scripts.
- ✅ **Pretrained Models:** Get started quickly with models trained for [microscopy](https://huggingface.co/talmolab/microscopy-pretrained) and [animal](https://huggingface.co/talmolab/animals-pretrained) domains. Available on Hugging Face.
- ✅ **Configurable Workflows:** Easily customize training and inference using YAML configuration files.
- ✅ **Visualization:** [Visualize](./visualizer.md) tracking results in your browser without any data leaving your machine, or use the SLEAP GUI for a more detailed view.
- ✅ **Examples:** Step-by-step notebooks and guides for common workflows.

## Installation

Head over to the [Installation Guide](./installation.md) to get started.


## Quickstart

Ready to try DREEM? Follow the [Quickstart Guide](./quickstart.md) to:

1. Download example datasets and pretrained models
2. Run tracking on sample videos
3. Visualize your results


## Example Workflows

Explore the Examples section for notebooks that walk you through the DREEM pipeline. We have an end-to-end demo that includes model training, as well as a microscopy example that shows how to use DREEM with an off-the-shelf detection model.


## Documentation Structure

- [Installation](./installation.md)
- [Quickstart](./quickstart.md)
- [Usage Guide](./usage.md)
- [Examples](./Examples/)
- [API Reference](https://dreem.sleap.ai/reference/dreem/)

## Get Help

- **Questions?** Open an issue on [GitHub](https://github.com/talmolab/dreem/issues).
- **Contributions:** We welcome contributions! See our [Contributing Guide](#) for details (link to be added).


<!-- ## Citing DREEM

If you use DREEM in your research, please cite our [paper](#) (link to be added). -->