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

- **Linux/Windows**: CUDA-enabled GPUs are supported via the `gpu` extra
- **macOS (Apple Silicon)**: MPS acceleration is used automatically

To install with GPU (CUDA) support:

```bash
uv pip install "dreem-track[gpu]"
```

Or with `uv tool`:

```bash
uv tool install "dreem-track[gpu]"
```

For a specific CUDA version, you can use `torch-cuda118`, `torch-cuda128`, or `torch-cuda130` instead of `gpu`.

## Development Installation

```bash
git clone https://github.com/talmolab/dreem && cd dreem
uv sync --extra gpu
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
