# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build, Lint, and Test

- **Run tests**: `pytest`
- **Run tests with coverage**: `pytest --cov=dreem --cov-report=xml tests/`
- **Lint code**: `black dreem tests`
- **Check linting**: `black --check dreem tests`
- **Check docstring style**: `pydocstyle --convention=google dreem/`

### Development Scripts

- **Train a model**: `dreem-train --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Run inference**: `dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Evaluate model**: `dreem-eval --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]`
- **Visualize results**: `dreem-visualize`

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
  - `microscopy_dataset.py` - For microscopy cell tracking
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

3. **Multi-anchor Crops**: For pose data, the system can crop around multiple body parts simultaneously to capture more spatial context.

4. **Flexible Visual Encoders**: Supports both timm and torchvision backends with easy swapping via config.

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