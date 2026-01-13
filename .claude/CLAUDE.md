# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL CONSTRAINTS
**NEVER** do the following:
- Do not automatically commit any changes you make to the repository until the user has provided approval. Always ask the user for approval before committing any changes. If the user has commands that specify how to commit changes, follow the steps of that command, but always ask the user for approval before committing any changes.
- Do not access any file containing credentials (e.g., `.env` files, `secrets/` directories).
- Do not leave empty placeholders in the code where the finished implementation should be.
- Do not flatter or give compliments unless specifically asked for judgment.
- Do not guess if you are unsure of my intent; always ask for clarification.
- Avoid generating new documentation unless explicitly requested; update existing docs instead

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for package management. All commands should be run with `uv run` prefix, or by running the script or command line tool after activating the virtual environment with `source .venv/bin/activate`.

### Setup

- **Install dependencies**: `uv sync --group dev`
- **Update lock file**: `uv lock`

### Build, Lint, and Test

- **Run tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov=dreem --cov-report=xml tests/`
- **Format code**: `uv run ruff format .`
- **Check formatting**: `uv run ruff format --check dreem tests`
- **Lint code**: `uv run ruff check .`
- **Fix lint issues**: `uv run ruff check --fix .`

### CLI Commands

The unified `dreem` CLI provides commands for training, tracking, and evaluation:

```bash
# Show help
dreem --help
dreem train --help
dreem track --help
dreem eval --help

# Run tracking inference
dreem track INPUT_DIR --checkpoint MODEL.ckpt --output RESULTS_DIR --crop-size 128

# Evaluate model against ground truth; in this case, INPUT_DIR should contain .slp files
# that have ground truth
dreem eval INPUT_DIR --checkpoint MODEL.ckpt --output RESULTS_DIR --crop-size 128

# Train a model
dreem train TRAIN_DIR --val-dir VAL_DIR --crop-size 128 --epochs 30
```

**Common flags:**
- `--gpu` / `-g`: Use GPU acceleration
- `--config` / `-c`: Override with a YAML config file
- `--set` / `-s`: Override individual config values (e.g., `--set tracker.max_tracks=5`)
- `--verbose`: Enable detailed logging
- `--quiet` / `-q`: Suppress progress output

## Architecture Overview

DREEM (Relates Every Entities' Motion) is a Global Tracking Transformer system for biological multi-object tracking. The codebase is organized around three main frameworks:

1. **Typer + Rich** - CLI framework with colored output and progress bars
2. **PyTorch** - Core model implementation and tensor operations
3. **PyTorch Lightning** - High-level training/validation/inference orchestration

### Core Components

- **`dreem/models/`** - Model architecture implementations
  - `global_tracking_transformer.py` - Main GTR model
  - `gtr_runner.py` - Lightning module wrapper for training/inference
  - `visual_encoder.py` - Visual feature extraction (supports timm/torchvision backends)
  - `transformer.py` - Custom transformer encoder/decoder implementation
  - `embedding.py` - Spatial and temporal positional embeddings

- **`dreem/datasets/`** - Data loading and preprocessing
  - `sleap_dataset.py` - For animal pose tracking with SLEAP annotations
  - `cell_tracking_dataset.py` - For Cell Tracking Challenge format data
  - `base_dataset.py` - Abstract base class for all datasets

- **`dreem/io/`** - Data structures and I/O utilities
  - `instance.py` - Core data structure for detected objects
  - `frame.py` - Frame-level data container
  - `track.py` - Trajectory representation
  - `association_matrix.py` - Tracking association logic

- **`dreem/inference/`** - Inference and tracking logic
  - `tracker.py` - Main tracking algorithm implementation
  - `track_queue.py` - Queue management for sliding window tracking

### Key Design Patterns

1. **Decoupled Detection and Tracking**: The system assumes pre-computed detections (from SLEAP, TrackMate, etc.) and focuses solely on associating them across time.

2. **Sliding Window Inference**: Tracking uses configurable window sizes to handle long videos efficiently while maintaining temporal context.

3. **Flexible Visual Encoders**: Supports both timm and torchvision backends with easy swapping via config.

### Configuration System

The CLI uses a layered configuration system with sensible defaults:

1. **Default configs** embedded in `dreem/configs/defaults/` (track.yaml, train.yaml)
2. **User config file** via `--config` flag (optional)
3. **CLI arguments** (highest priority)
4. **`--set` overrides** for fine-grained control. This is not recommended and a config file should be used instead. It is only provided for maximum control.

Key config sections:
- `model` - Model architecture parameters
- `dataset` - Data loading settings
- `optimizer`/`scheduler` - Training optimization
- `tracker` - Inference-time tracking parameters
- `loss` - Loss function configuration

Configs support hierarchical overrides via `--set` using dot notation (e.g., `--set model.nhead=8`).

## GitHub Workflow
- Always use the `gh` CLI to do GitHub related tasks like opening PRs, inspecting issues, and anything that interacts with GitHub. Always prefer using the `gh` CLI over fetching the entire page if URLs with `https://github.com` are provided as part of your task.

## Ways of working

- When implementing a new feature, create a new directory named ./{feature_name}, where feature_name is an appropriate name for the feature. In this directory, write out a to-do list, and as  you complete your work, update that to-do list so that anyone can refer to current progress at any time 
