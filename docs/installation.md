# Installation

The easiest way to install DREEM is with [uv](https://docs.astral.sh/uv/). 

In a new directory:

```bash
uv venv && source .venv/bin/activate
uv pip install dreem-track
```

Or install as a standalone tool without creating a virtual environment:

```bash
uv tool install dreem-track
```

You can also use pip:

```bash
pip install dreem-track
```

## GPU Support

DREEM automatically uses GPU acceleration when available:

- **Linux/Windows**: CUDA-enabled GPUs are supported out of the box
- **macOS (Apple Silicon)**: MPS acceleration is used automatically

No additional configuration is needed.

## Development Installation

```bash
git clone https://github.com/talmolab/dreem && cd dreem
uv sync --extra torch-cuda128 
```

## Verify Installation

Check that DREEM is installed correctly:

```bash
dreem --help
```

## Uninstall

```bash
uv pip uninstall dreem-track
# or
uv tool uninstall dreem-track
```
