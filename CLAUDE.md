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

This project uses [uv](https://docs.astral.sh/uv/) for package management. All commands should be run with `uv run` prefix, or after activating the virtual environment with `source .venv/bin/activate`.

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

### Development Scripts

- **Train a model**: `uv run dreem-train --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Run inference**: `uv run dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Evaluate model**: `uv run dreem-eval --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Visualize**: `uv run dreem-visualize`

## Architecture Overview

DREEM (Relates Every Entities' Motion) is a Global Tracking Transformer system for biological multi-object tracking. The codebase is organized around three main frameworks:

1. **Hydra** - Configuration management system used throughout for handling YAML configs
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
  - `batch_tracker.py` - Batched tracking for efficiency
  - `track_queue.py` - Queue management for sliding window tracking

### Key Design Patterns

1. **Decoupled Detection and Tracking**: The system assumes pre-computed detections (from SLEAP, TrackMate, etc.) and focuses solely on associating them across time.

2. **Sliding Window Inference**: Tracking uses configurable window sizes to handle long videos efficiently while maintaining temporal context.

3. **Flexible Visual Encoders**: Supports both timm and torchvision backends with easy swapping via config.

### Configuration System

All training/inference parameters are managed through Hydra configs. Key config sections:
- `model` - Model architecture parameters
- `dataset` - Data loading settings
- `optimizer`/`scheduler` - Training optimization
- `tracker` - Inference-time tracking parameters
- `loss` - Loss function configuration

Configs support hierarchical overrides via CLI using dot notation (e.g., `model.nhead=8`).

## GitHub Workflow
- Always use the `gh` CLI to do GitHub related tasks like opening PRs, inspecting issues, and anything that interacts with GitHub. Always prefer using the `gh` CLI over fetching the entire page if URLs with `https://github.com` are provided as part of your task.
