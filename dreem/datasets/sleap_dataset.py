"""Module containing logic for loading sleap datasets."""

import logging
import os
import random
from math import inf
from pathlib import Path
from typing import Optional, Union

import albumentations as A
import imageio
import numpy as np
import sleap_io as sio
import torch
from torchvision.transforms import functional as tvf

from dreem.datasets import BaseDataset, data_utils
from dreem.datasets.preprocessors import NonMaxSuppression, RemoveExcessDetections
from dreem.io import Frame, Instance

logger = logging.getLogger("dreem.datasets")


def resolve_identity_carrier(labels: sio.Labels) -> str:
    """Return which sleap-io field carries the association label for `labels`.

    sleap-io (>=0.9.0) separates three things `Track` used to be overloaded for:

    - `Track`    -- a trajectory contiguous in time *within one video*
    - `Identity` -- one physical animal, recognizable across videos and sessions
    - `Category` -- a class several distinct animals can share

    For association training the label we want is the *widest* one available: an
    `Identity` is the same animal everywhere, so it pairs detections across videos
    that no `Track` could ever pair. So the carrier is `"identity"` when the file
    carries any, and `"track"` otherwise.

    The carrier is resolved once per file and applied uniformly, rather than
    per-instance. Choosing per instance would mix two *independent* index spaces --
    `identities.index(...)` and `tracks.index(...)` both start at 0 -- so identity 0
    and track 0 would silently become the same `gt_track_id`, i.e. two different
    animals asserted to be one. Uniform-per-file makes that unrepresentable.

    Args:
        labels: The labels file to inspect.

    Returns:
        Either `"identity"` or `"track"`.
    """
    for lf in labels:
        for inst in lf.instances:
            if getattr(inst, "identity", None) is not None:
                return "identity"
    return "track"


def identity_index(instance, labels: sio.Labels, carrier: str) -> int:
    """Return the association label index for `instance` under `carrier`.

    Args:
        instance: The sleap-io instance to label.
        labels: The labels file it came from (supplies the index registry).
        carrier: `"identity"` or `"track"`, from `resolve_identity_carrier`.

    Returns:
        The index of the instance's carrier value in the corresponding registry,
        or `-1` if the instance carries no value for it.

    Notes:
        `-1` means "unlabeled", and every unlabeled detection in a frame gets the
        same `-1`. Downstream that reads as *one* identity, so several unlabeled
        animals in one frame are asserted to be the same animal. That predates this
        change; `SleapDataset` now counts and warns rather than leaving it silent.
    """
    if carrier == "identity":
        identity = getattr(instance, "identity", None)
        if identity is None:
            return -1
        return labels.identities.index(identity)
    if instance.track is None:
        return -1
    return labels.tracks.index(instance.track)


class SleapDataset(BaseDataset):
    """Dataset for loading animal behavior data from sleap."""

    def __init__(
        self,
        slp_files: list[str],
        video_files: list[str],
        data_dirs: Optional[list[str]] = None,
        padding: int = 5,
        crop_size: Union[int, list[int]] = 128,
        anchors: int | list[str] | str = "",
        chunk: bool = True,
        clip_length: int = 16,
        mode: str = "train",
        handle_missing: str = "centroid",
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
        verbose: bool = False,
        normalize_image: bool = True,
        max_batching_gap: int = 15,
        use_tight_bbox: bool = False,
        dilation_radius_px: Union[int, list[int]] = 0,
        max_detection_overlap: float = 0,
        max_tracks: int = inf,
        burst_aug: bool = False,
        n_bursts: int = 4,
        burst_max_span: int = 900,
        feature_cache: str | None = None,
        skip_image: bool = False,
        bbox_mode: str | None = None,
        **kwargs,
    ):
        """Initialize SleapDataset.

        Args:
            slp_files: a list of .slp files storing tracking annotations
            video_files: a list of paths to video files
            data_dirs: a path, or a list of paths to data directories. If provided, crop_size should be a list of integers
                with the same length as data_dirs.
            padding: amount of padding around object crops
            crop_size: the size of the object crops. Can be either:
                - An integer specifying a single crop size for all objects
                - A list of integers specifying different crop sizes for different data directories
            anchors: One of:
                        * a string indicating a single node to center crops around
                        * a list of skeleton node names to be used as the center of crops
                        * an int indicating the number of anchors to randomly select
                    If unavailable then crop around the midpoint between all visible anchors.
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train`, `val`, or `test`. Determines whether this dataset is used for
                training, validation/testing/inference.
            handle_missing: how to handle missing single nodes. one of `["drop", "ignore", "centroid"]`.
                            if "drop" then we dont include instances which are missing the `anchor`.
                            if "ignore" then we use a mask instead of a crop and nan centroids/bboxes.
                            if "centroid" then we default to the pose centroid as the node to crop around.
            augmentations: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augmentations = {
                        'Rotate': {'limit': [-90, 90], 'p': 0.5},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0, 'p': 0.2},
                        'RandomContrast': {'limit': 0.2, 'p': 0.6}
                    }
                Spatial augmentations (e.g. rotations, flips, affine warps) are
                drawn once per clip and applied to every frame in it, so instances
                keep coherent trajectories; pixel-level ones are drawn per frame.
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            verbose: boolean representing whether to print
            normalize_image: whether to normalize the image to [0, 1]
            max_batching_gap: the max number of frames that can be unlabelled before starting a new batch
            use_tight_bbox: whether to use tight bounding box (around keypoints) instead of the default square bounding box
                Equivalent to `bbox_mode="tight"` with no padding; `bbox_mode` wins if both are set.
            dilation_radius_px: radius of the keypoints dilation in pixels. 0 means no mask applied
            max_detection_overlap: the iom threshold for non-maximum suppression of detections
            max_tracks: the maximum number of tracks that can be created while tracking. Remove any detections that exceed this number.
            burst_aug: sample each chunk as `n_bursts` runs of consecutive frames
                spread over a log-uniform span, instead of one contiguous run.
                Puts short-range and long-range pairs in the SAME window so the
                association loss must tell the two regimes apart. See
                `BaseDataset._burst_chunks`.
            n_bursts: number of consecutive-frame bursts per chunk (burst_aug only).
            burst_max_span: largest frame span a chunk may cover (burst_aug only).
                Keep it within one clip (900 frames here) so every gap is honest.
            feature_cache: path to a .npz written by `precompute_features.py`. When
                set, each instance gets its precomputed embedding attached and its
                crop left empty, so `GlobalTrackingTransformer.extract_features`
                skips the visual encoder entirely (it only computes for instances
                without features). Only valid with a FROZEN encoder -- cached
                vectors cannot follow a moving one -- and it disables crop
                augmentation, since a cached feature cannot vary per epoch.
            **kwargs: Additional keyword arguments (unused but accepted for compatibility)
            skip_image: never decode frames. Each instance gets a 1x1 placeholder
                crop and the frame size comes from the video metadata, for models
                whose visual encoder ignores pixels (`encoder_type: none`).
                Spatial augmentations still move the keypoints: they run on a
                blank frame of the true size, so rotations turn about the same
                center as with real images. Pixel-level ones have nothing to act on.
            bbox_mode: how each anchor's box is fit to the pose. One of
                * "square" (default): a `crop_size` square on the anchor, shifted
                  (not clipped) to start at 0 when it crosses the top or left
                  edge, so near those edges the box is no longer centered.
                * "square_centered": a `crop_size` square always centered on
                  the anchor, even past an edge.
                * "tight": the keypoints' axis-aligned extent.
                All three add `padding` on every side. A single-node pose falls
                back to "square_centered" under "tight", which has no extent.
                None keeps the legacy behavior: "tight" if `use_tight_bbox`, else
                "square" (padding is not applied to legacy tight boxes).
        """
        # These must be set BEFORE super().__init__(): it calls create_chunks(),
        # which reads them through getattr on self.
        self.burst_aug = burst_aug
        self.n_bursts = n_bursts
        self.burst_max_span = burst_max_span

        # Frozen-encoder feature cache. Keys are (labels-file stem, frame index,
        # gt track id) -- all three derived through this same loader when the
        # cache was written, so they line up by construction.
        self.feature_cache = feature_cache
        self._features = None
        if feature_cache is not None:
            import numpy as _np

            _z = _np.load(feature_cache, allow_pickle=False)
            _keys, _vecs = _z["keys"], _z["features"]
            self._features = {
                (str(k[0]), int(k[1]), int(k[2])): v for k, v in zip(_keys, _vecs)
            }
            logger.info(
                f"feature cache: {len(self._features):,} vectors from {feature_cache}"
            )

        super().__init__(
            slp_files,
            video_files,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
        )

        # spatial augmentations are drawn once per clip, not once per frame
        self.clip_augmentations = (
            data_utils.ClipAugmentation(self.augmentations)
            if self.augmentations is not None
            else None
        )

        self.slp_files = slp_files
        self.data_dirs = data_dirs
        self.video_files = video_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.mode = mode.lower()
        self.handle_missing = handle_missing.lower()
        if self._features is not None and self.augmentations is not None:
            # A cached feature is one fixed vector per instance; it cannot follow
            # a crop that is re-augmented every epoch. Silently skipping the
            # augmentations would train on a different distribution than the
            # config describes, so refuse instead.
            raise ValueError(
                "feature_cache cannot be combined with crop augmentations: a "
                "cached feature is computed once and cannot vary per epoch. Set "
                "dataset.<split>.augmentations to null, or drop feature_cache."
            )

        self.n_chunks = n_chunks
        self.seed = seed
        self.normalize_image = normalize_image
        self.max_batching_gap = max_batching_gap
        self.use_tight_bbox = use_tight_bbox
        self.skip_image = skip_image
        if bbox_mode not in (None, "square", "square_centered", "tight"):
            raise ValueError(
                "bbox_mode must be one of 'square', 'square_centered', 'tight' "
                f"or None, got {bbox_mode!r}"
            )
        self.bbox_mode = bbox_mode
        self.dilation_radius_px = dilation_radius_px
        self.max_detection_overlap = (
            max_detection_overlap if max_detection_overlap is not None else 0
        )
        self.max_tracks = max_tracks if max_tracks is not None else inf
        if isinstance(anchors, int):
            self.anchors = anchors
        elif isinstance(anchors, str):
            self.anchors = [anchors]
        else:
            self.anchors = anchors

        if not isinstance(self.data_dirs, list):
            self.data_dirs = [self.data_dirs]

        if not isinstance(self.crop_size, list):
            # make a list so its handled consistently if multiple crops are used
            if len(self.data_dirs) > 0:  # for test mode, data_dirs is []
                self.crop_size = [self.crop_size] * len(self.data_dirs)
            else:
                self.crop_size = [self.crop_size]

        if not isinstance(self.dilation_radius_px, list):
            self.dilation_radius_px = [self.dilation_radius_px] * len(self.data_dirs)
        else:
            self.dilation_radius_px = [self.dilation_radius_px]

        if len(self.data_dirs) > 0 and len(self.crop_size) != len(self.data_dirs):
            raise ValueError(
                f"If a list of crop sizes or data directories are given,"
                f"they must have the same length but got {len(self.crop_size)} "
                f"and {len(self.data_dirs)}"
            )

        if (
            isinstance(self.anchors, list) and len(self.anchors) == 0
        ) or self.anchors == 0:
            raise ValueError(f"Must provide at least one anchor but got {self.anchors}")

        self.verbose = verbose

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        # load_slp is a wrapper around sio.load_slp for frame gap checks
        self.labels = []
        self.annotated_segments = {}
        # Which sleap-io field supplies gt_track_id, per labels file. Parallel to
        # `self.labels`; see `resolve_identity_carrier`.
        self.identity_carriers = []
        for slp_file in self.slp_files:
            labels, annotated_segments = data_utils.load_slp(slp_file)
            self.labels.append(labels)
            self.annotated_segments[slp_file] = annotated_segments
            carrier = resolve_identity_carrier(labels)
            self.identity_carriers.append(carrier)
            n_missing = sum(
                identity_index(inst, labels, carrier) < 0
                for lf in labels
                for inst in lf.instances
            )
            logger.info(
                f"{Path(slp_file).name}: associating on sio.{carrier.capitalize()} "
                f"({len(labels.identities if carrier == 'identity' else labels.tracks)} "
                f"distinct)."
            )
            if n_missing:
                logger.warning(
                    f"{Path(slp_file).name}: {n_missing} instance(s) carry no "
                    f"sio.{carrier.capitalize()} and share gt_track_id=-1, so they "
                    f"will train as ONE animal. Label them, or drop them, if that "
                    f"is not what you mean."
                )

        for vid_file in self.vid_files:
            if not Path(vid_file).exists():
                raise FileNotFoundError(f"Video file not found: {vid_file}")
        # Video readers are opened lazily, once per process (see
        # `_get_video_reader`). An imageio reader wraps an ffmpeg subprocess, so a
        # reader opened here and inherited by forked DataLoader workers would have
        # every worker reading the same pipe, handing crops to the wrong frames.
        self.videos = [None] * len(self.vid_files)
        self._videos_pid = os.getpid()
        self._inherited_videos = []
        # preprocessors
        self.remove_excess_detections = RemoveExcessDetections(max_tracks)
        self.non_max_suppression = NonMaxSuppression(max_detection_overlap)
        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks_slp()

    def get_indices(self, idx: int) -> tuple:
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def _cache_stem(self, label_idx: int) -> str:
        """Labels-file stem used as the feature cache's first key component."""
        from pathlib import Path as _P

        return _P(str(self.label_files[label_idx])).stem

    def get_instances(
        self, label_idx: list[int], frame_idx: torch.Tensor
    ) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: indices of the frames to load in to the batch

        Returns:
            A list of `dreem.io.Frame` objects containing metadata and instance data for the batch/clip.

        """
        sleap_labels_obj = self.labels[label_idx]
        carrier = self.identity_carriers[label_idx]
        video_name = self.video_files[label_idx]

        # get the correct crop size based on the video
        video_par_path = Path(video_name).parent
        if len(self.data_dirs) > 0:
            crop_size = self.crop_size[0]
            dilation_radius_px = self.dilation_radius_px[0]
            for j, data_dir in enumerate(self.data_dirs):
                if Path(data_dir) == video_par_path:
                    crop_size = self.crop_size[j]
                    dilation_radius_px = self.dilation_radius_px[j]
                    break
        else:
            crop_size = self.crop_size[0]
            dilation_radius_px = self.dilation_radius_px[0]

        vid_reader = self._get_video_reader(label_idx)

        skeleton = sleap_labels_obj.skeletons[-1]

        frames = []
        max_crop_h, max_crop_w = 0, 0
        aug_replay = None  # spatial augmentation params shared by the whole clip
        for i, frame_ind in enumerate(frame_idx):
            (
                instances,
                gt_track_ids,
                poses,
                shown_poses,
                point_scores,
                instance_score,
            ) = ([], [], [], [], [], [])

            frame_ind = int(frame_ind)

            # sleap-io method for indexing a Labels() object based on the frame's index
            lf = sleap_labels_obj[(sleap_labels_obj.video, frame_ind)]
            if frame_ind != lf.frame_idx:
                logger.warning(f"Frame index mismatch: {frame_ind} != {lf.frame_idx}")

            if self._features is not None or self.skip_image:
                # Cached features mean no crop is ever needed, so skip the decode.
                # This is the expensive half under burst sampling: the frames in a
                # chunk are scattered, so every one is a random seek into the mp4.
                # Frame size still has to come from somewhere -- take it from the
                # video's metadata rather than a decoded array.
                img = None
                _shape = self.labels[label_idx].video.shape
                h, w = int(_shape[1]), int(_shape[2])
                c = int(_shape[3]) if len(_shape) > 3 else 1
            else:
                try:
                    img = vid_reader.get_data(int(frame_ind))
                except IndexError as e:
                    logger.warning(
                        f"Could not read frame {frame_ind} from {video_name} due to {e}"
                    )
                    continue

            if img is not None and len(img.shape) == 2:
                img = img.expand_dims(-1)
            if img is not None:
                h, w, c = img.shape

            if img is not None and c == 1:
                img = np.concatenate(
                    [img, img, img], axis=-1
                )  # convert to grayscale to rgb

            if img is not None and np.issubdtype(img.dtype, np.integer):
                img = img.astype(np.float32)
                if self.normalize_image:
                    img = img / 255

            n_instances_dropped = 0

            gt_instances = []
            # don't load instances that have been 'greyed out' i.e. all nans for keypoints
            for inst in lf.instances:
                pts = np.array([p for p in inst.numpy()])
                if np.isnan(pts).all():
                    continue
                else:
                    gt_instances.append(inst)

            dict_instances = {}
            no_track_instances = []
            for instance in gt_instances:
                gt_track_id = identity_index(instance, sleap_labels_obj, carrier)
                if gt_track_id >= 0:
                    if gt_track_id not in dict_instances:
                        dict_instances[gt_track_id] = instance
                    else:
                        existing_instance = dict_instances[gt_track_id]
                        # if existing is PredictedInstance and current is not, then current is a UserInstance and should be used
                        if isinstance(
                            existing_instance, sio.PredictedInstance
                        ) and not isinstance(instance, sio.PredictedInstance):
                            dict_instances[gt_track_id] = instance
                else:
                    no_track_instances.append(instance)

            gt_instances = list(dict_instances.values()) + no_track_instances

            if self.mode == "train":
                np.random.shuffle(gt_instances)

            for instance in gt_instances:
                if (
                    np.random.uniform() < self.instance_dropout["p"]
                    and n_instances_dropped < self.instance_dropout["n"]
                ):
                    n_instances_dropped += 1
                    continue

                gt_track_ids.append(
                    identity_index(instance, sleap_labels_obj, carrier)
                )

                poses.append(
                    dict(
                        zip(
                            [n.name for n in instance.skeleton.nodes],
                            [p for p in instance.numpy()],
                        )
                    )
                )

                shown_poses = [
                    {
                        key: val
                        for key, val in instance.items()
                        if not np.isnan(val).any()
                    }
                    for instance in poses
                ]

                point_scores.append(
                    np.array(
                        [
                            (
                                1.0  # point scores not reliably available in sleap io PredictedPointsArray
                                # point.score
                                # if isinstance(point, sio.PredictedPoint)
                                # else 1.0
                            )
                            for point in instance.numpy()
                        ]
                    )
                )
                if isinstance(instance, sio.PredictedInstance):
                    instance_score.append(instance.score)
                else:
                    instance_score.append(1.0)
            # augmentations
            if self.augmentations is not None:
                for transform in self.augmentations:
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                if shown_poses:
                    keypoints = np.vstack([list(s.values()) for s in shown_poses])

                else:
                    keypoints = []

                # skip_image: spatial transforms need the frame's size to move
                # keypoints, not its pixels, so hand them a blank frame.
                aug_img = img if not self.skip_image else np.zeros((h, w, 1), np.uint8)
                augmented, aug_replay = self.clip_augmentations(
                    image=aug_img, keypoints=keypoints, replay=aug_replay
                )

                aug_poses = augmented["keypoints"]
                if not self.skip_image:
                    img = augmented["image"]

                aug_poses = [
                    arr
                    for arr in np.split(
                        np.array(aug_poses),
                        np.array([len(s) for s in shown_poses]).cumsum(),
                    )
                    if arr.size != 0
                ]

                aug_poses = [
                    dict(zip(list(pose_dict.keys()), aug_pose_arr.tolist()))
                    for aug_pose_arr, pose_dict in zip(aug_poses, shown_poses)
                ]

                _ = [
                    pose.update(aug_pose)
                    for pose, aug_pose in zip(shown_poses, aug_poses)
                ]

            if img is not None:
                img = tvf.to_tensor(img)

            for j in range(len(gt_track_ids)):
                pose = shown_poses[j]

                """Check for anchor"""
                crops = []
                boxes = []
                centroids = {}

                if isinstance(self.anchors, int):
                    anchors_to_choose = list(pose.keys()) + ["midpoint"]
                    anchors = np.random.choice(anchors_to_choose, self.anchors)
                else:
                    anchors = self.anchors

                dropped_anchors = self.node_dropout(anchors)

                for anchor in anchors:
                    if anchor in dropped_anchors:
                        centroid = np.array([np.nan, np.nan])

                    elif anchor == "midpoint" or anchor == "centroid":
                        centroid = np.nanmean(np.array(list(pose.values())), axis=0)

                    elif anchor in pose:
                        centroid = np.array(pose[anchor])
                        if np.isnan(centroid).any():
                            centroid = np.array([np.nan, np.nan])

                    elif (
                        anchor not in pose
                        and len(anchors) == 1
                        and self.handle_missing == "centroid"
                    ):
                        anchor = "midpoint"
                        centroid = np.nanmean(np.array(list(pose.values())), axis=0)

                    else:
                        centroid = np.array([np.nan, np.nan])

                    arr_pose = np.array(list(pose.values()))

                    if np.isnan(centroid).all():
                        bbox = torch.tensor([np.nan, np.nan, np.nan, np.nan])
                    elif self.bbox_mode is not None:
                        bbox = self._fit_bbox(centroid, arr_pose, crop_size)
                    else:
                        if self.use_tight_bbox and len(pose) > 1:
                            # tight bbox, dont allow this for centroid-only poses!
                            # note bbox will be a different size for each instance; padded at the end of the loop
                            bbox = data_utils.get_tight_bbox(arr_pose)

                        else:
                            bbox = data_utils.pad_bbox(
                                data_utils.get_bbox(centroid, crop_size),
                                padding=self.padding,
                            )

                    if self.skip_image:
                        # 1x1 stand-in; the null encoder never reads it
                        crop = torch.zeros(1, 1, 1)
                    elif img is None:
                        # cached features: bbox/centroid are still needed by the
                        # positional embedding, the pixels are not
                        crop = torch.empty(0)
                    elif bbox.isnan().all():
                        crop = torch.zeros(
                            c,
                            crop_size + 2 * self.padding,
                            crop_size + 2 * self.padding,
                            dtype=img.dtype,
                        )
                    else:
                        crop = data_utils.crop_bbox(img, bbox)

                    if dilation_radius_px > 0:
                        if np.isnan(arr_pose).any():
                            logger.warning("arr_pose is nan")
                        mask = data_utils.get_mask_from_keypoints(
                            arr_pose, crop, dilation_radius_px, bbox
                        )
                        crop = crop * mask
                        # logger.debug(f"Applying mask to crop {frame_ind}_{j}")

                    crops.append(crop)
                    # get max h,w for padding for tight bboxes
                    # empty when features come from the cache; skip_image placeholders
                    # must not overwrite the frame size h, w read from metadata
                    if crop.ndim == 3 and not self.skip_image:
                        c, h, w = crop.shape
                        if h > max_crop_h:
                            max_crop_h = h
                        if w > max_crop_w:
                            max_crop_w = w

                    centroids[anchor] = centroid
                    boxes.append(bbox)

                if self.skip_image:
                    crops = crops[0]  # one placeholder per instance, any anchor count
                elif len(crops) > 0 and img is not None:
                    crops = torch.concat(crops, dim=0)
                elif img is None:
                    crops = torch.empty(0)

                if len(boxes) > 0:
                    boxes = torch.stack(boxes, dim=0)

                if self.handle_missing == "drop" and boxes.isnan().any():
                    continue

                instance = Instance(
                    gt_track_id=gt_track_ids[j],
                    pred_track_id=-1,
                    crop=crops,
                    centroid=centroids,
                    bbox=boxes,
                    skeleton=skeleton,
                    pose=poses[j],
                    point_scores=point_scores[j],
                    instance_score=instance_score[j],
                )

                if self._features is not None:
                    key = (self._cache_stem(label_idx), frame_ind,
                           int(gt_track_ids[j]))
                    vec = self._features.get(key)
                    if vec is None:
                        raise KeyError(
                            f"feature cache has no entry for {key}. Rebuild it with "
                            "precompute_features.py against this exact dataset -- a "
                            "stale cache would silently train on the wrong crops."
                        )
                    instance.features = torch.as_tensor(vec)

                instances.append(instance)

            # remove excess detections
            if len(instances) > self.max_tracks:
                state = self.remove_excess_detections.run(
                    {
                        "frame_ind": frame_ind,
                        "instances": instances,
                    }
                )
                instances = state["instances"]

            # non-maximum suppression (high overlap bounding boxes)
            if self.max_detection_overlap > 0 and len(instances) > 0:
                state = self.non_max_suppression.run(
                    {
                        "frame_ind": frame_ind,
                        "instances": instances,
                    }
                )
                instances = state["instances"]

            frame = Frame(
                video_id=label_idx,
                frame_id=frame_ind,
                vid_file=video_name,
                # (C, H, W); from metadata when the decode was skipped
                img_shape=(img.shape if img is not None
                           else torch.Size([c, h, w])),
                instances=instances,
            )
            frames.append(frame)

        return frames

    def _fit_bbox(
        self, anchor_xy: np.ndarray, pose_xy: np.ndarray, crop_size: int
    ) -> torch.Tensor:
        """Fit one anchor's (y1, x1, y2, x2) box according to `self.bbox_mode`.

        Args:
            anchor_xy: (x, y) of the anchor.
            pose_xy: (n_visible_nodes, 2) visible keypoints of the instance.
            crop_size: side of the square boxes, in pixels.

        Returns:
            The padded box in pixels.
        """
        mode = self.bbox_mode
        if mode == "tight" and len(pose_xy) < 2:
            mode = "square_centered"
        if mode == "square":
            bbox = data_utils.get_bbox(anchor_xy, crop_size)
        elif mode == "square_centered":
            cx, cy = float(anchor_xy[0]), float(anchor_xy[1])
            half = crop_size / 2
            bbox = torch.Tensor([cy - half, cx - half, cy + half, cx + half])
        else:
            bbox = data_utils.get_tight_bbox(pose_xy)
        return data_utils.pad_bbox(bbox, padding=self.padding)

    def _get_video_reader(self, label_idx: int) -> "imageio.core.Format.Reader":
        """Get the video reader for a labels file, opened by this process.

        A forked DataLoader worker inherits the parent's readers, which share the
        parent's ffmpeg subprocesses. On the first call in a new process those are
        set aside, unused, and the worker opens its own. They are kept referenced
        rather than closed: closing one would terminate the ffmpeg process the
        parent is still reading from.

        Args:
            label_idx: index of the labels file (and its video).

        Returns:
            An imageio reader for the video, owned by the calling process.
        """
        pid = os.getpid()
        if self._videos_pid != pid:
            self._inherited_videos.extend(r for r in self.videos if r is not None)
            self.videos = [None] * len(self.vid_files)
            self._videos_pid = pid
        if self.videos[label_idx] is None:
            self.videos[label_idx] = imageio.get_reader(self.vid_files[label_idx])
        return self.videos[label_idx]

    def __getstate__(self) -> dict:
        """Drop open video readers when pickled, e.g. for spawned workers.

        Returns:
            The dataset's state, with readers to be reopened on first use.
        """
        state = self.__dict__.copy()
        state["videos"] = [None] * len(self.vid_files)
        state["_videos_pid"] = None
        state["_inherited_videos"] = []
        return state

    def __del__(self):
        """Close the video readers this process opened before garbage collection."""
        if getattr(self, "_videos_pid", None) != os.getpid():
            return  # inherited readers belong to the parent process
        for reader in getattr(self, "videos", []):
            if reader is not None:
                reader.close()
