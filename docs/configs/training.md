# Training Configuration

This guide describes the parameters used for training configuration. Most parameters can be set via CLI flags (see `dreem train --help`) or through a YAML config file.

> **Note**: For defaults, leave fields blank or omit the key. Use `null` to disable features like logging or early stopping.

## Model Configuration

The `model` section configures the transformer architecture and visual encoder.

### Core Model Parameters

* `ckpt_path` (`str` | `null`): Path to model checkpoint file for resuming training (use `null` for new training).
* `d_model` (`int`): Size of embedding dimensions for the transformer (default: 128).
* `nhead` (`int`): Number of attention heads in transformer encoder/decoder layers (default: 1).
* `num_encoder_layers` (`int`): Number of layers in transformer encoder (default: 1).
* `num_decoder_layers` (`int`): Number of layers in transformer decoder (default: 1).
* `dropout` (`float`): Dropout probability used in each transformer layer (default: 0.1).
* `activation` (`str`): Activation function. Options: `"relu"`, `"gelu"`, `"glu"` (default: `"relu"`).
* `return_intermediate_dec` (`bool`): Whether to return output from intermediate decoder layers (default: `true`).
* `norm` (`bool`): Whether to normalize output of encoder and decoder (default: `false`).
* `num_layers_attn_head` (`int`): Number of layers in the AttentionHead block (default: 1).
* `dropout_attn_head` (`float`): Dropout probability for the AttentionHead block (default: 0.1).
* `return_embedding` (`bool`): Whether to return spatiotemporal embeddings (default: `false`).
* `decoder_self_attn` (`bool`): Whether to use self attention in the decoder (default: `true`).

### Embedding Configuration (`embedding_meta`)

#### Positional Embedding (`pos`)

* `mode` (`str`): Embedding type. Options: `"fixed"` (sinusoidal), `"learned"`, or `null` (no embedding).
* `n_points` (`int`): Number of points to embed (1 for centroid, 3+ for bbox coordinates).
* `normalize` (`bool`): Whether to normalize positions (only for fixed embeddings, default: `true`).
* `temperature` (`float`): Temperature constant for sinusoidal position embedding (only for fixed, default: 10000).
* `scale` (`float` | `null`): Factor to scale positions after normalizing (only for fixed, optional).
* `emb_num` (`int`): Number of embeddings in lookup table (only for learned).
* `over_boxes` (`bool`): Whether to compute embedding for bbox coordinates (`y1x1y2x2`) or centroid+size (`yxwh`) (only for learned, default: `true`).

**MLP Configuration** (`mlp_cfg`): Required when `n_points > 1`, optional otherwise.

* `hidden_dims` (`int`): Dimensionality of MLP hidden layers.
* `num_layers` (`int`): Number of hidden layers.
* `dropout` (`float`): Dropout probability for each hidden layer.

#### Temporal Embedding (`temp`)

* `mode` (`str`): Embedding type. Options: `"fixed"`, `"learned"`, or `null` (no embedding).
* `temperature` (`float`): Temperature constant for sinusoidal embedding (only for fixed, default: 10000).
* `emb_num` (`int`): Number of embeddings in lookup table (only for learned).

### Visual Encoder Configuration (`encoder_cfg`)

* `model_name` (`str`): Name of visual encoder backbone. For `timm` backend, any model in `timm.list_models()` is available. For `torchvision`, only ResNet models are available.
* `backend` (`str`): Backend library. Options: `"timm"` or `"torchvision"` (default: `"timm"`).
* `in_chans` (`int`): Number of input channels (default: 3, use more for multi-anchor crops).
* `pretrained` (`bool`): Whether to use pretrained weights or initialize randomly (default: `false`).

> **Note**: For advanced users, see [`timm.create_model`](https://timm.fast.ai/create_model) or [`torchvision.models.resnet`](https://pytorch.org/vision/stable/models/resnet.html) for additional parameters.

## Loss Configuration

The `loss` section configures the Association Loss function.

* `neg_unmatched` (`bool`): Whether to set unmatched objects to background (default: `false`).
* `epsilon` (`float`): Small value for numerical precision to prevent division by zero (default: `1e-4`).
* `asso_weight` (`float`): Weight for association loss (default: 1.0).

## Optimizer Configuration

The `optimizer` section configures the training optimizer.

* `name` (`str`): Optimizer name (must match PyTorch optimizer class name exactly, case-sensitive). See [`torch.optim`](https://pytorch.org/docs/stable/optim.html#algorithms) for available options.

**Adam Parameters** (default optimizer):

* `lr` (`float`): Learning rate (default: 0.0001).
* `betas` (`tuple[float, float]`): Coefficients for computing running averages of gradient and its square (default: `[0.9, 0.999]`).
* `eps` (`float`): Term added to denominator for numerical stability (default: `1e-8`).
* `weight_decay` (`float`): L2 penalty (default: 0.01).

> **Note**: For other optimizers, see the respective PyTorch documentation for available parameters.

## Scheduler Configuration

The `scheduler` section configures the learning rate scheduler.

* `name` (`str`): Scheduler name (must match PyTorch scheduler class name exactly, case-sensitive). See [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for available options.

**ReduceLROnPlateau Parameters** (default scheduler):

* `mode` (`str`): One of `"min"` or `"max"`. In `min` mode, LR reduces when monitored quantity stops decreasing.
* `factor` (`float`): Factor by which LR is reduced: `new_lr = lr * factor` (default: 0.5).
* `patience` (`int`): Number of epochs with no improvement before reducing LR (default: 5).
* `threshold` (`float`): Threshold for measuring new optimum to focus on significant changes (default: 0.001).
* `threshold_mode` (`str`): One of `"rel"` or `"abs"`. Relative or absolute threshold mode (default: `"rel"`).

## Dataset Configuration

The `dataset` section configures training, validation, and optionally test datasets. Requires `train_dataset` and optionally `val_dataset` and `test_dataset` keys.

### Directory-based Input (Recommended)

Use the `dir` section to automatically discover videos and labels:

* `path` (`str`): Path to directory containing videos and labels (use absolute paths).
* `labels_suffix` (`str`): File extension for label files (e.g., `.slp`, `.csv`, `.xml`).
* `vid_suffix` (`str`): File extension for video files (e.g., `.mp4`, `.avi`, `.tif`, `.tiff`).

### File-based Input

Alternatively, specify files explicitly:

* `slp_files` (`list[str]`): List of paths to SLEAP label files (`.slp`).
* `video_files` (`list[str]`): List of paths to video files.

### Dataset Parameters

* `crop_size` (`int`): Size (in pixels) of the square bounding box around each instance. Should match approximate size of tracked objects.
* `clip_length` (`int`): Number of frames per chunk when processing videos (default: 32).
* `chunk` (`bool`): Whether to chunk videos into smaller clips (default: `true`).
* `anchors` (`str` | `list[str]` | `int`):
  * String: Single node name to center crops around (e.g., `"centroid"`).
  * List: Multiple node names to use as crop centers.
  * Integer: Number of anchors to randomly select.
* `padding` (`int`): Amount of padding added to each side of the bounding box (default: 0).
* `mode` (`str`): Dataset mode. Options: `"train"` or `"val"` (determines usage).
* `handle_missing` (`str`): How to handle missing anchor nodes. Options: `"drop"`, `"ignore"`, or `"centroid"` (only for SleapDataset).

### Augmentations

The `augmentations` subsection contains parameters for [Albumentations](https://albumentations.ai). Keys must match augmentation class names exactly. Additional augmentations include `NodeDropout` and `InstanceDropout`.

**Example**:
```YAML
augmentations:
  Rotate:
    limit: 45
    p: 0.3
  MotionBlur:
    blur_limit: [3, 7]
    p: 0.3
```

> **Note**: Augmentations are typically only used for training datasets, not validation or test.

## Dataloader Configuration

The `dataloader` section configures data loading. Should have `train_dataloader` and optionally `val_dataloader`/`test_dataloader` keys.

* `shuffle` (`bool`): Whether to reshuffle data at every epoch. Should be `true` for training, `false` for validation/test (default: `false`).
* `num_workers` (`int`): Number of subprocesses for data loading. Use `0` for single-process loading.

> **Note**: For advanced options, see [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

## Logging Configuration

The `logging` section configures experiment logging.

* `logger_type` (`str` | `null`): Logger to use. Options: `"CSVLogger"`, `"TensorBoardLogger"`, `"WandbLogger"`, or `null` to disable.

**WandbLogger Parameters** (recommended):

* `name` (`str`): Short display name for this run (default: `"dreem_train"`).
* `project` (`str`): Name of the project where runs are sent.
* `entity` (`str`): Username or team name where runs are sent.
* `save_dir` (`str`): Absolute path to directory where metadata is stored.
* `log_model` (`str`): Log checkpoints as W&B artifacts. Options: `"all"`, `"best"`, or `null`.
* `group` (`str`): Group name to organize runs into larger experiments.
* `notes` (`str`): Longer description of the run.

> **Note**: See [`wandb.init()`](https://docs.wandb.ai/ref/python/init) and [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html) for more parameters.

## Early Stopping Configuration

The `early_stopping` section configures early stopping for training.

* `monitor` (`str`): Quantity to monitor (e.g., `"val_loss"`).
* `mode` (`str`): One of `"min"` or `"max"`. In `min` mode, stops when quantity stops decreasing.
* `patience` (`int`): Number of checks with no improvement before stopping (default: 10).
* `min_delta` (`float`): Minimum change to qualify as improvement (default: 0.1).
* `check_finite` (`bool`): Stop training when monitor becomes NaN or infinite (default: `true`).
* `stopping_threshold` (`float` | `null`): Stop immediately when monitor reaches this threshold (optional).
* `divergence_threshold` (`float` | `null`): Stop when monitor becomes worse than this threshold (optional).

## Checkpointing Configuration

The `checkpointing` section configures model checkpointing.

* `monitor` (`list[str]`): List of metrics to save best models for. Usually `"val_{METRIC}"` notation. A separate ModelCheckpoint is created for each metric.
* `dirpath` (`str` | `null`): Directory to save models. If `null`, saves to `./models/[GROUP]/[NAME]` or `./models/[NAME]` (default: `null`).
* `save_last` (`bool`): Save a `last.ckpt` copy whenever a checkpoint is saved (default: `true`).
* `save_top_k` (`int`): Save the best k models. Use `-1` to save all, `0` to save none (default: `-1`).
* `every_n_epochs` (`int`): Number of epochs between checkpoints. Set to `0` to disable periodic saves (default: 1).

## Trainer Configuration

The `trainer` section configures the PyTorch Lightning Trainer.

* `accelerator` (`str`): Device type. Options: `"cpu"`, `"gpu"`, `"cuda"`, `"tpu"`, `"ipu"`, `"hpu"`, `"mps"`, `"auto"`.
* `strategy` (`str` | `null`): Training strategy (e.g., `"ddp"`, `"deepspeed"`). Optional.
* `devices` (`list[int]` | `int` | `str`): Device indices to use. Can be a number, list, `-1` for all devices, or `"auto"`.
* `max_epochs` (`int`): Stop training after this many epochs. Use `-1` for infinite training (default: 20).
* `min_epochs` (`int`): Force training for at least this many epochs (default: 1).
* `check_val_every_n_epoch` (`int`): Perform validation loop every N training epochs (default: 1).
* `enable_checkpointing` (`bool`): Enable checkpointing (default: `true`).
* `gradient_clip_val` (`float` | `null`): Value at which to clip gradients (optional).
* `limit_train_batches` (`float` | `int`): Fraction or number of training batches to process (default: 1.0).
* `limit_val_batches` (`float` | `int`): Fraction or number of validation batches to process (default: 1.0).
* `log_every_n_steps` (`int`): How often to log within steps (default: 1).

> **Note**: For advanced options, see [`lightning.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html).


## Example Configuration

```YAML
# Model configuration
model:
  # str | null: Path to checkpoint for resuming training (null for new training)
  ckpt_path: <str | null>
  
  # int: Size of embedding dimensions (default: 128)
  d_model: <int>
  
  # int: Number of attention heads (default: 1)
  nhead: <int>
  
  # int: Number of encoder layers (default: 1)
  num_encoder_layers: <int>
  
  # int: Number of decoder layers (default: 1)
  num_decoder_layers: <int>
  
  # float: Dropout probability (default: 0.1)
  dropout: <float>
  
  # str: Activation function: "relu", "gelu", or "glu" (default: "relu")
  activation: <str>
  
  # bool: Normalize encoder/decoder output (default: false)
  norm: <bool>
  
  # int: Number of layers in AttentionHead (default: 1)
  num_layers_attn_head: <int>
  
  # float: Dropout for AttentionHead (default: 0.1)
  dropout_attn_head: <float>
  
  # bool: Use self attention in decoder (default: true)
  decoder_self_attn: <bool>
  
  # Embedding configuration
  embedding_meta:
    # Positional embedding
    pos:
      # str: Embedding mode: "fixed", "learned", or null
      mode: <str | null>
      
      # int: Number of points to embed (1 for centroid, 3+ for bbox)
      n_points: <int>
      
      # bool: Normalize positions (only for fixed, default: true)
      normalize: <bool>
      
      # float: Temperature for sinusoidal embedding (only for fixed, default: 10000)
      temperature: <float>
      
      # float | null: Scale factor after normalizing (only for fixed, optional)
      scale: <float | null>
      
      # int: Number of embeddings in lookup table (only for learned)
      emb_num: <int>
      
      # bool: Embed bbox coordinates vs centroid+size (only for learned, default: true)
      over_boxes: <bool>
      
      # MLP config (required when n_points > 1)
      mlp_cfg: <dict | null>
        # int: Hidden layer dimensionality
        hidden_dims: <int>
        # int: Number of hidden layers
        num_layers: <int>
        # float: Dropout probability
        dropout: <float>
    
    # Temporal embedding
    temp:
      # str: Embedding mode: "fixed", "learned", or null
      mode: <str | null>
      
      # float: Temperature for sinusoidal embedding (only for fixed, default: 10000)
      temperature: <float>
      
      # int: Number of embeddings in lookup table (only for learned)
      emb_num: <int>
  
  # Visual encoder configuration
  encoder_cfg:
    # str: Model name (e.g., "resnet18")
    model_name: <str>
    
    # str: Backend: "timm" or "torchvision" (default: "timm")
    backend: <str>
    
    # int: Number of input channels (default: 3)
    in_chans: <int>
    
    # bool: Use pretrained weights (default: false)
    pretrained: <bool>

# Loss configuration
loss:
  # bool: Set unmatched objects to background (default: false)
  neg_unmatched: <bool>
  
  # float: Small value for numerical precision (default: 1e-4)
  epsilon: <float>
  
  # float: Weight for association loss (default: 1.0)
  asso_weight: <float>

# Optimizer configuration
optimizer:
  # str: Optimizer name (must match PyTorch class name exactly)
  name: <str>
  
  # float: Learning rate (default: 0.0001)
  lr: <float>
  
  # list[float]: Beta coefficients (default: [0.9, 0.999])
  betas: <list[float]>
  
  # float: Epsilon for numerical stability (default: 1e-8)
  eps: <float>
  
  # float: Weight decay (L2 penalty, default: 0.01)
  weight_decay: <float>

# Scheduler configuration
scheduler:
  # str: Scheduler name (must match PyTorch class name exactly)
  name: <str>
  
  # str: Mode: "min" or "max"
  mode: <str>
  
  # float: Factor by which LR is reduced (default: 0.5)
  factor: <float>
  
  # int: Patience before reducing LR (default: 5)
  patience: <int>
  
  # float: Threshold for measuring improvement (default: 0.001)
  threshold: <float>
  
  # str: Threshold mode: "rel" or "abs" (default: "rel")
  threshold_mode: <str>

# Dataset configuration
dataset:
  train_dataset:
    # int: Number of chunks to train on (dataset size is clip_length * n_chunks)
    n_chunks: <int>
    
    # int: Size of mask around keypoint (in pixels) to mask out background (default: 0; no masking)
    dilation_radius_px: <int>
    
    # str: Anchor node for centering crops
    anchors: <str>
    
    # int: Number of frames per chunk (default: 32)
    clip_length: <int>
    
    # int: Size of bounding box in pixels; if list, should be same length as dir.path (e.g., [crop_size_1, crop_size_2] for two datasets)
    crop_size: <int | list[int]>
    
    # int: Padding added to each side of bbox (default: 0)
    padding: <int>
    
    # Directory-based input (recommended)
    dir:
      # str: Path or list of paths to directories with data (use absolute paths)
      # e.g. ["/path/to/dataset1", "/path/to/dataset2"]
      path: <str | list[str]>
      
      # str: File extension for label files
      labels_suffix: <str>
      
      # str: File extension for video files
      vid_suffix: <str>
    
    # dict: Augmentations (typically only for training)
    augmentations: <dict>
      # Example augmentations (keys must match Albumentations class names)
    #   GaussianBlur:
    #     blur_limit:
    #     - 3
    #     - 7
    #     p: 0.3
    #     sigma_limit: 0.1
      # Rotate:
      #   limit: <int>
      #   p: <float>
      # MotionBlur:
      #   blur_limit: <list[int]>
      #   p: <float>
  
  val_dataset:
    # Same structure as train_dataset (typically no augmentations)
    ...

# Dataloader configuration
dataloader:
  train_dataloader:
    # bool: Shuffle data (should be true for training, default: false)
    shuffle: <bool>

  val_dataloader:
    # bool: Shuffle data (should be false for validation, default: false)
    shuffle: <bool>

# Logging configuration
logging:
  # str | null: Logger type: "CSVLogger", "TensorBoardLogger", "WandbLogger", or null
  logger_type: <str | null>
  
  # str: Display name for this run (default: "dreem_train")
  name: <str>
  
  # str: Project name (for WandbLogger)
  project: <str | null>
  
  # str: Entity/team name (for WandbLogger)
  entity: <str | null>
  
  # bool: Log model checkpoints (default: true)
  log_model: <bool>
  
  # str: Group name for organizing runs
  group: <str>

# Early stopping configuration
early_stopping:
  # see Pytorch Lightning Early Stopping documentation

# Checkpointing configuration
checkpointing:
  # list[str]: Metrics to monitor for best models (e.g., ["val_loss"])
  monitor: <list[str]>
  
  # str | null: Directory to save models (null = auto)
  dirpath: <str | null>
  
  # bool: Save last checkpoint (default: true)
  save_last: <bool>
  
  # int: Save top k models (-1 = all, 0 = none, default: -1)
  save_top_k: <int>
  
  # int: Save every N epochs (default: 1)
  every_n_epochs: <int>

# Trainer configuration
trainer:
  # str: Device type: "cpu", "gpu", "cuda", etc.
  accelerator: <str>
  
  # str | null: Training strategy (optional)
  strategy: <str | null>
  
  # list[int] | int | str: Device indices
  devices: <list[int] | int | str>
  
  # int: Maximum number of epochs (default: 20, -1 = infinite)
  max_epochs: <int>
  
  # int: Minimum number of epochs (default: 1)
  min_epochs: <int>
  
  # int: Validate every N epochs (default: 1)
  check_val_every_n_epoch: <int>
  
  # bool: Enable checkpointing (default: true)
  enable_checkpointing: <bool>
  
  # float | int: Fraction or number of training batches (default: 1.0)
  limit_train_batches: <float | int>
  
  # float | int: Fraction or number of validation batches (default: 1.0)
  limit_val_batches: <float | int>
  
  # int: Log every N steps (default: 1)
  log_every_n_steps: <int>
```

## CLI Usage

Most parameters can be set via CLI flags. For example:

```bash
dreem train ./data/train \
  --val-dir ./data/val \
  --crop-size 128 \
  --epochs 20 \
  --lr 0.0001 \
  --d-model 128 \
  --nhead 1 \
  --encoder-layers 1 \
  --decoder-layers 1 \
  --anchor centroid \
  --clip-length 32 \
```

See `dreem train --help` for all available CLI options.

