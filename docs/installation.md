# Installation

Install DREEM into a fresh environment, add GPU support if you have one, and check that the `dreem` command works. DREEM runs on Linux, Windows and macOS. A GPU is optional, and the [Quickstart](quickstart.md) runs on a CPU.

## 1. Install DREEM

=== "uv (recommended)"

    In a new folder:

    ```bash
    uv venv && source .venv/bin/activate
    uv pip install dreem-track
    ```

    On Windows (PowerShell), activate with `.venv\Scripts\activate` instead. No uv yet? See [uv's installation guide](https://docs.astral.sh/uv/getting-started/installation/).

=== "uv tool"

    Installs the `dreem` command on its own, without a virtual environment:

    ```bash
    uv tool install dreem-track
    ```

=== "pip"

    In an environment of your choice:

    ```bash
    pip install dreem-track
    ```

## 2. Add GPU support (optional)

DREEM uses a GPU automatically when it finds one; `--device` chooses one explicitly.

=== "NVIDIA GPU (Linux, Windows)"

    Install the CUDA build of PyTorch with the `gpu` extra:

    ```bash
    uv pip install "dreem-track[gpu]"
    ```

    With `uv tool`, use `uv tool install "dreem-track[gpu]"`. For a specific CUDA version, use the `torch-cuda118`, `torch-cuda128` or `torch-cuda130` extra instead of `gpu`.

=== "Apple Silicon"

    Nothing to add: DREEM uses Apple's MPS acceleration automatically.

=== "CPU only"

    Nothing to add. Tracking runs on the CPU, more slowly than on a GPU.

## 3. Check that it worked

```bash
dreem --help
```

!!! success "You should see"
    The list of commands:

    ```text
    ╭─ Commands ───────────────────────────────────────────────────────────────────╮
    │ track     Run tracking on a dataset.                                         │
    │ eval      Evaluate a trained DREEM model against ground truth.               │
    │ convert   Convert external tracking formats to .slp files.                   │
    │ train     Train a DREEM model.                                               │
    │ render    Render tracking results as a video with colored overlays.          │
    │ system    Display system information and GPU status.                         │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ```

To check whether DREEM sees your GPU, run `dreem system`.

??? failure "If it fails"
    - **`dreem: command not found`**: the environment is not active. Run `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) in the folder where you created it, or use `uv tool install`.
    - **No GPU in `dreem system`**: install the `gpu` extra (step 2), and check that `nvidia-smi` works on your machine.

Next: the [Quickstart](quickstart.md) tracks a sample video in a few minutes.

## Development installation

To work on DREEM itself:

```bash
git clone https://github.com/talmolab/dreem && cd dreem
uv sync --extra gpu
```

## Uninstall

```bash
uv pip uninstall dreem-track
# or, if installed with uv tool
uv tool uninstall dreem-track
```
