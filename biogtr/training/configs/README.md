# Description of training parameters

Here, we describe the hyperparameters used for setting up training

## `model`

This section contains all the parameters for initializing a `GTRRunner` object

* `ckpt_path`: `str` containing the path to model `.ckpt` file. Used for resuming training.
* `d_model`: an `int` indicating the size of the embedding dimensions used for input into the transformer
* `nhead`: an `int` indicating the number of attention heads used in the transformer's encoder/decoder layers.
* `num_encoder_layers`: an `int` indicating the number of layers in the transformer encoder block
* `num_decoder_layers`: an `int` indicating the number of layers in the transformer decoder block
* `dropout`: a `float` indicating the dropout probability used in each transformer layer
* `activation`: One of {`"relu"`, `"gelu"` `"glu"`}. Which activation function to use in the transformer
* `return_intermediate_dec`: A `bool` indicating whether or not to return the output from the intermediate decoder layers.
* `norm`: A `bool` indicating whether or not to normalize output of encoder and decoder.
* num_layers_attn_head: An `int` indicating The number of layers in the `AttentionHead` block.
* `dropout_attn_head`: A `float `representing the dropout probability for the `AttentionHead` block.
* `return_embedding`: A `bool` indicating whether to return the positional embeddings
* `decoder_self_attn`: A `bool` indicating whether to use self attention in the decoder.
### `embedding_meta`: 

This section contains parameters for initializing the `Embedding` Layer.

#### `pos`

This subsection contains the parameters for initializing a Spatial `Embedding`.

* mode: One of {`"fixed"`, `"learned"`, `"None"`}. Indicates whether to use a fixed sinusoidal, learned, or no embedding.

Note: See `biogtr.models.Embedding` for additional `kwargs` that can be passed

#### `temp`

This subsection contains the parameters for initializing a Temporal `Embedding`

* mode: One of {`"fixed"`, `"learned"`, `"None"`}. Indicates whether to use a fixed sinusoidal, learned, or no embedding.

Note: See `biogtr.models.Embedding` for additional `kwargs` that can be passed
            
### `encoder_cfg`

This section contains all the parameters for initializing a `VisualEncoder` model.

* `model_name`: `str` name of the visual encoder backbone to be used. When using `timm` as a backend, all models in `timm.list_model` are available. However, when using `torchvision` as a backend, only `resnet`s are available for now.

* `backend`: Either `timm` or `torchvision`. Indicates which deep learning library to use for initializing the visual encoder

* `in_chans`: an `int` indicating the number of input channels input images contain. Mostly used for multi-anchor crops

Note: See [`timm.create_model`](https://timm.fast.ai/create_model) or [`torchvision.models.resnet`](https://pytorch.org/vision/stable/models/resnet.html) for additional `kwargs` that can be passed to the visual encoder.

## `loss`

This section contains parameters for the Association Loss function

* `neg_unmatched` a bool indicating whether to set unmatched objects to the background
* `epsilon`: A small `float` used for numeric precision to prevent dividing by zero
* `asso_weight`: A `float` indicating how much to weight the association loss by

## `optimizer`

This section contains the parameters for initializing the training optimizer

* name: A `str` representation of the optimizer. See [`torch.optim`](https://pytorch.org/docs/stable/optim.html#algorithms) for available optimizers. `name` must match the optimizer name exactly (case-sensitive).

Note: For arguments needed to initialize your requested optimizer, please see the respective pytorch documentation page.

## `scheduler` 

This section contains parameters for initializing the learning rate scheduler.

* name: A `str` representation of the scheduler. See [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for available schedulers. `name` must match the scheduler name exactly (case-sensitive).

Note: For arguments needed to initialize your requested scheduler, please see the respective pytorch documentation page.

## `tracker`:

This section contains parameters for initializing the `Tracker`

* `window_size`: the size of the window used during sliding inference.
* `use_vis_feats`: Whether or not to use visual feature extractor.
* `overlap_thresh`: the trajectory overlap threshold to be used for assignment.
* `mult_thresh`: Whether or not to use weight threshold.
* `decay_time`: weight for `decay_time` postprocessing.
* `iou`: Either `{None, '', "mult" or "max"}`. Whether to use multiplicative or max iou reweighting.
* `max_center_dist`: distance threshold for filtering trajectory score matrix.
* `persistent_tracking`: whether to keep a buffer across chunks or not.
* `max_gap`: the max number of frames a trajectory can be missing before termination.
* `max_tracks`: the maximum number of tracks that can be created while tracking.
We force the tracker to assign instances to a track instead of creating a new track if `max_tracks `has been reached.

## `runner`

This section contains parameters for how to handle training/validation/testing

### `metrics`

This section contains config for which metrics to compute during training/validation/testing. See [`pymotmetrics.list_metrics`](https://github.com/cheind/py-motmetrics) for available metrics.

Should have a `train`, `val` and `test` key with corresponding list of metrics to compute during training.

### `persistent_tracking`

This section indicates whether or not to track across chunks during training/validation/testing

Should have a `train`, `val` and `test` key with a corresponding `bool` indicating whether to use persistent tracking.

### `dataset`

This section contains the params for initializing the datasets for training. Requires a `train_dataset` and optionally `val_dataset`, `test_dataset` keys. For the correct args corresponding to the labels and vid_files paths as well as specific `kwargs` please see the corresponding `biogtr.datasets` submodule. For instance `SleapDataset` requires a `slp_files` and `video_files` key while `MicroscopyDataset` requires `videos` and `tracks`. Below we describe the common parameters regardless of dataset

* `padding`: An `int` representing the amount of padding to be added to each side of the bounding box size
* `crop_size`: Either an `int` or `tuple` indicating the size of the bounding box around which a crop will form.
* `chunk`: Whether or not to chunk videos into smaller clips to feed to model
* `clip_length`: the number of frames in each chunk
* `mode`: `train` or `val`. Determines whether this dataset is used for training or validation.
* `n_chunks`: Number of chunks to subsample from. Can either a fraction of the dataset (ie `(0,1.0]`) or number of chunks
* `seed`: set a seed for reproducibility
* `gt_list`: An optional path to .txt file containing ground truth for cell tracking challenge datasets.

#### `dir`:
This section allows you to pass a directory rather than paths to labels/videos individually

* `path`: The path to the dir where the data is stored (recommend absolute path)
* `labels_suffix`: A `str` containing the file extension to search for labels files. e.g. `.slp`, `.csv`, or `.xml`.
* `vid_suffix`: A `str` containing the file extension to search for video files e.g `.mp4`, `.avi` or `.tif`.
#### `augmentations`:

This subsection contains params for albumentations. See [`albumentations`](https://albumentations.ai) for available visual augmentations. Other available augmentations include `NodeDropout` and `InstanceDropout`. Keys must match augmentation class name exactly and contain subsections with parameters for the augmentation e.g.

```YAML
augmentations: 
    Rotate:
        limit: 45
        p: 0.3
    ...
    MotionBlur:
        blur_limit: [3,7]
        p: 0.3
```

### `dataloader`

This section outlines the params needed for the dataloader. Should have a `train_dataloader` and optionally `val_dataloader`/`test_dataloader` keys. See [`torch.utils.data.Dataloader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for available args.

### `logging`:

This section sets up logging for the training job. 

* `logger_type`: A string indicating which logger to use. Available loggers are {`"CSVLogger"`, `"TensorBoardLogger"`,`"WandbLogger"`}

Note: Please see the documentation for the corresponding logger at [`lightning.loggers`](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) for available parameters.

### `early_stopping`

This section configures early stopping for training runs. See [`lightning.callbacks.EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping) for available arguments

### `checkpointing`

This section enables model checkpointing during training

* `monitor`: A list of metrics to save best models for. Usually should be `"val_{METRIC}"` notation

See [`lightning.callbacks.ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) for available checkpointing params and generally more info on how `lightning` sets up checkpoints

### `trainer`

This section configures the `lightning.Trainer` object for training. Please see `lightning.Trainer`(https://lightning.ai/docs/pytorch/stable/common/trainer.html) for available arguments and how the `trainer` works in general

### `view_batch`

This section allows you to visualize the data before training

* `enable`: A `bool` indicating whether or not to view a batch
* `num_frames`: An `int` indicating The number of frames in the batch to visualize
* `no_train`: A `bool` indicating whether or not to train after visualization is complete


