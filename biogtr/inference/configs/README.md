# Description of inference params

Here we describe the parameters used for inference.

* `ckpt_path`: A `str` containing the path to the saved model checkpoint. Can optionally provide a list of models and this will trigger batch inference where each pod gets a model to run inference with.

* `out_dir`: A `str` containing a directory path where to store outputs.
## `tracker`

This section configures the tracker.

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

## `dataset`:
This section contains the params for initializing the datasets for training. Requires a `test_dataset` keys.
### `test_dataset`:
For the correct args corresponding to the labels and vid_files paths as well as specific `kwargs` please see the corresponding `biogtr.datasets` submodule. For instance `SleapDataset` requires a `slp_files` and `video_files` key while `MicroscopyDataset` requires `videos` and `tracks`. Below we describe the common parameters regardless of dataset

* `padding`: An `int` representing the amount of padding to be added to each side of the bounding box size
* `crop_size`: Either an `int` or `tuple` indicating the size of the bounding box around which a crop will form.
* `chunk`: Whether or not to chunk videos into smaller clips to feed to model
* `clip_length`: the number of frames in each chunk
* `mode`: `train` or `val`. Determines whether this dataset is used for training or validation. Should always set this to `test` or `val`.
* `seed`: set a seed for reproducibility
* `gt_list`: An optional path to .txt file containing ground truth for cell tracking challenge datasets.

#### `dir`:
This section allows you to pass a directory rather than paths to labels/videos individually

* `path`: The path to the dir where the data is stored (recommend absolute path)
* `labels_suffix`: A `str` containing the file extension to search for labels files. e.g. `.slp`, `.csv`, or `.xml`.
* `vid_suffix`: A `str` containing the file extension to search for video files e.g `.mp4`, `.avi` or `.tif`.