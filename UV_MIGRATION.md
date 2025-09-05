# DREEM UV Migration Guide

This document describes the migration from conda-based installation to uv-based installation for the DREEM project.

## What Changed

### 1. Updated `pyproject.toml`
- **Added all missing dependencies** from conda environment files:
  - `matplotlib`
  - `opencv-python < 4.9.0`
  - `ffmpeg-python`
  - `motmetrics`
  - `seaborn`
  - `wandb`
  - `timm`
  - `huggingface-hub[cli]`

- **Added platform-specific optional dependency groups**:
  - `cuda`: For CUDA-enabled systems (Linux/Windows with GPU)
  - `cpu`: For CPU-only systems (Linux/Windows)
  - `apple-silicon`: For Apple Silicon (macOS)

- **Fixed package discovery** by explicitly specifying `packages = ["dreem"]`

### 2. Created Installation Scripts
- **`install.sh`**: Interactive installation script for Linux/macOS
- **`install.bat`**: Interactive installation script for Windows
- Both scripts automatically detect the platform and ask for user preferences

### 3. Updated Documentation
- **README.md**: Added comprehensive uv installation instructions
- **Platform-specific installation options**: Clear instructions for each platform
- **Legacy conda support**: Maintained for backward compatibility

## Installation Methods

### Quick Installation (Recommended)
```bash
# Linux/macOS
./install.sh

# Windows
install.bat
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/talmolab/dreem && cd dreem

# Platform-specific installation
uv sync --extra cuda        # CUDA-enabled systems
uv sync --extra cpu         # CPU-only systems  
uv sync --extra apple-silicon  # Apple Silicon

# Development installation
uv sync --extra dev --extra cuda  # or cpu/apple-silicon
```

### Using the Environment
```bash
# Option 1: Direct execution
uv run dreem-train --help

# Option 2: Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Platform Support

The migration maintains full cross-platform compatibility:

- **Linux/Windows with CUDA**: `uv sync --extra cuda`
- **Linux/Windows CPU-only**: `uv sync --extra cpu`
- **macOS (Apple Silicon)**: `uv sync --extra apple-silicon`

## Benefits of UV Migration

1. **Faster installation**: uv is significantly faster than conda
2. **Better dependency resolution**: More reliable dependency management
3. **Simpler workflow**: Single tool for Python package management
4. **Cross-platform consistency**: Same commands work across all platforms
5. **Lock file support**: Reproducible builds with `uv.lock`

## Backward Compatibility

- **Conda environments preserved**: All original `.yml` files remain unchanged
- **Legacy installation**: Users can still use conda if preferred
- **Documentation updated**: Both installation methods documented

## Testing

All installation methods have been tested:
- ✅ `uv sync --extra cuda --dry-run`
- ✅ `uv sync --extra cpu --dry-run`
- ✅ `uv sync --extra apple-silicon --dry-run`
- ✅ `uv sync --extra dev --extra apple-silicon --dry-run`
- ✅ Installation script execution

## Migration Complete

The project is now fully compatible with uv installation while maintaining backward compatibility with conda. Users can choose their preferred installation method.
