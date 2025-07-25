# Description of inference params

Here we describe the parameters used for inference. See [here](./eval.md#example-config) for an example inference config.

* `ckpt_path`: (`str`) the path to the saved model checkpoint. Can optionally provide a list of models and this will trigger batch inference where each pod gets a model to run inference with.
e.g:
```YAML
...
ckpt_path: "/path/to/model.ckpt"
...
```
## `tracker`

This section configures the tracker.

* `window_size`: (`int`) the size of the window used during sliding inference.
* `use_vis_feats`: (`bool`) Whether or not to use visual feature extractor.
* `overlap_thresh`: (`float`) the trajectory overlap threshold to be used for assignment.
* `mult_thresh`: (`bool`) Whether or not to use weight threshold.
* `decay_time`: (`float`) weight for `decay_time` postprocessing.
* `iou`: (`str` | `None`) Either `{None, '', "mult" or "max"}`. Whether to use multiplicative or max iou reweighting.
* `max_center_dist`: (`float`) distance threshold for filtering trajectory score matrix.
* `persistent_tracking`: (`bool`) whether to keep a buffer across chunks or not.
* `max_gap`: (`int`) the max number of frames a trajectory can be missing before termination.
* `max_tracks`: (`int`) the maximum number of tracks that can be created while tracking.
    We force the tracker to assign instances to a track instead of creating a new track if `max_tracks `has been reached.

### Examples:
```YAML
...
tracker:
    window_size: 8
    overlap_thresh: 0.01
    mult_thresh: false
    decay_time: 0.9
    iou: "mult"
    max_center_dist: 0.1
...
```

## `dataset`
This section contains the params for initializing the datasets for training. Requires a `test_dataset` keys. 

### [`BaseDataset`](../reference/dreem/datasets/base_dataset.md) args

* `padding`: An `int` representing the amount of padding to be added to each side of the bounding box size
* `crop_size`: (`int`|`tuple`) the size of the bounding box around which a crop will form.
* `chunk`: Whether or not to chunk videos into smaller clips to feed to model
* `clip_length`: the number of frames in each chunk
* `mode`: `train` or `val`. Determines whether this dataset is used for training or validation.
* `n_chunks`: Number of chunks to subsample from. Can either a fraction of the dataset (ie `(0,1.0]`) or number of chunks
* `seed`: set a seed for reproducibility
* `gt_list`: An optional path to .txt file containing ground truth for cell tracking challenge datasets.

#### `dir`:
This section allows you to pass a directory rather than paths to labels/videos individually

* `path`: The path to the dir where the data is stored (recommend absolute path)
* `labels_suffix`: (`str`) containing the file extension to search for labels files. e.g. `.slp`, `.csv`, or `.xml`.
* `vid_suffix`: (`str`) containing the file extension to search for video files e.g `.mp4`, `.avi` or `.tif`.
##### Examples:
```YAML
...
dataset:
    ...
    {MODE}_dataset:
        dir:
            path: "/path/to/data/dir/mode"
            labels_suffix: ".slp"
            vid_suffix: ".mp4"
        ...
    ...
...
```
#### `augmentations`:

This subsection contains params for albumentations. See [`albumentations`](https://albumentations.ai) for available visual augmentations. Other available augmentations include `NodeDropout` and `InstanceDropout`. Keys must match augmentation class name exactly and contain subsections with parameters for the augmentation

##### Example
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
### [`SleapDataset`](../reference/dreem/datasets/sleap_dataset.md) Args:
* `slp_files`: (`str`) a list of .slp files storing tracking annotations
* `video_files`: (`str`) a list of paths to video files
* `anchors`: (`str` | `list` | `int`) One of:
    * a string indicating a single node to center crops around
    * a list of skeleton node names to be used as the center of crops
    * an int indicating the number of anchors to randomly select
    If unavailable then crop around the midpoint between all visible anchors.
* `handle_missing`: how to handle missing single nodes. one of [`"drop"`, `"ignore"`, `"centroid"`].
    * if `drop` then we dont include instances which are missing the `anchor`.
    * if `ignore` then we use a mask instead of a crop and nan centroids/bboxes.
    * if `centroid` then we default to the pose centroid as the node to crop around.
### [`MicroscopyDataset`](../reference/dreem/datasets/microscopy_dataset.md)
* `videos`: (`list[str | list[str]]`) paths to raw microscopy videos
* `tracks`: (`list[str]`) paths to trackmate gt labels (either `.xml` or `.csv`)
* `source`: file format of gt labels based on label generator. Either `"trackmate"` or `"isbi"`.
### [`CellTrackingDataset`](../reference/dreem/datasets/cell_tracking_dataset.md)
* `raw_images`: (`list[list[str] | list[list[str]]]`) paths to raw microscopy images
* `gt_images`: (`list[list[str] | list[list[str]]]`) paths to gt label images
* `gt_list`: (`list[str]`) An optional path to .txt file containing gt ids stored in cell
                tracking challenge format: `"track_id", "start_frame",
                "end_frame", "parent_id"`
### `dataset` Examples
#### [`SleapDataset`](../reference/dreem/datasets/sleap_dataset.md)
```YAML
...
dataset:
    test_dataset:
        slp_files: ["/path/to/test/labels1.slp", "/path/to/test/labels2.slp", ..., "/path/to/test/labelsN.slp"]
        video_files: ["/path/to/test/video1.mp4", "/path/to/test/video2.mp4", ..., "/path/to/test/videoN.mp4"]
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        anchors: ["node1", "node2", ..."node_n"]
        handle_missing: "drop"
        ... # we don't include augmentations bc usually you shouldn't use augmentations during val/test
...
```
#### [`MicroscopyDataset`](../reference/dreem/datasets/microscopy_dataset.md)
```YAML
dataset:
    test_dataset:
        tracks: ["/path/to/test/labels1.csv", "/path/to/test/labels2.csv", ..., "/path/to/test/labelsN.csv"]
        videos: ["/path/to/test/video1.tiff", "/path/to/test/video2.tiff", ..., "/path/to/test/videoN.tiff"]
        source: "trackmate"
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        ... # we don't include augmentations bc usually you shouldn't use augmentations during val/test
```

## dataloader
This section outlines the params needed for the dataloader. Should have a `test_dataloader` 
> Below we list the args we found useful/necessary for the dataloaders. For more advanced users see [`torch.utils.data.Dataloader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for more ways to initialize the dataloaders

* `shuffle`: (`bool`) Set to `True` to have the data reshuffled at every epoch (during training, this should always be `True` and during val/test usually `False`) 
* `num_workers`: (`int`) How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

### Example
```YAML
...
dataloader:
    test_dataloader: # we leave out the `shuffle` field as default=`False` which is what we want
        num_workers: 4
...
```
## `runner`
This section outlines arguments to be overridden for the GTR Runner
* `save_path`: Path to `*.hdf5` file where eval results will be saved
* `metrics`: Contains a subkey called `test` with a list of pymotmetrics to be computed or `"all"` to compute all metrics
### Example (Computing all metrics):
```YAML
runner:
    save_path: "./test_eval.hdf5"
    metrics:
        test: "all"
```
### Example (Only computing `num_switches`)
```YAML
runner:
    save_path: "./test_eval.hdf5"
    metrics:
        test: ["num_switches"]
```
## Example Config

```YAML
--8<-- "dreem/inference/configs/eval.yaml"
```