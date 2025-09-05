# Installation

## Using uv (Recommended)

### Prerequisites
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repository:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```

### Quick installation (Recommended):
Use our automated installation script:

**Linux/macOS:**
```bash
./install.sh
```

**Windows:**
```cmd
install.bat
```

### Manual platform-specific installation:

#### Linux/Windows with CUDA (GPU-accelerated):
```bash
uv sync --extra cuda
```

#### Linux/Windows CPU-only:
```bash
uv sync --extra cpu
```

#### macOS (Apple Silicon):
```bash
uv sync --extra apple-silicon
```

#### Development installation (includes dev dependencies):
```bash
uv sync --extra dev --extra cuda  # or --extra cpu or --extra apple-silicon
```

### Activate the environment:
```bash
# Option 1: Use uv run to execute commands directly
uv run dreem-train --help

# Option 2: Activate the virtual environment manually
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Uninstalling
```bash
rm -rf .venv     # Remove the virtual environment
rm uv.lock       # Remove the lock file (optional)
```

## Using conda (Legacy)

### Clone the repository:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```

### Set up in a new conda environment:
#### Linux/Windows:
##### GPU-accelerated (requires CUDA/nvidia gpu)
```bash
conda env create -f environment.yml && conda activate dreem
```
##### CPU:
```bash
conda env create -f environment_cpu.yml && conda activate dreem
```
#### OSX (Apple Silicon)
```bash
conda env create -f environment_osx-arm64.yml && conda activate dreem
```

### Uninstall
```bash
conda env remove -n dreem
```