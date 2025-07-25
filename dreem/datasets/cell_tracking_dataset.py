"""Module containing cell tracking challenge dataset."""

import random
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import sleap_io as sio
import torch
from PIL import Image
from scipy.ndimage import measurements

from dreem.datasets import BaseDataset, data_utils
from dreem.io import Frame, Instance


class CellTrackingDataset(BaseDataset):
    """Dataset for loading cell tracking challenge data."""

    def __init__(
        self,
        gt_list: list[list[str]],
        raw_img_list: list[list[str]],
        data_dirs: Optional[list[str]] = None,
        padding: int = 5,
        crop_size: int = 20,
        chunk: bool = False,
        clip_length: int = 10,
        mode: str = "train",
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
        max_batching_gap: int = 15,
        use_tight_bbox: bool = False,
        ctc_track_meta: list[str] | None = None,
        apply_mask_to_crop: bool = False,
        **kwargs,
    ):
        """Initialize CellTrackingDataset.

        Args:
            gt_list: filepaths of gt label images in a list of lists (each list
                corresponds to a dataset)
            raw_img_list: filepaths of original tif images in a list of lists
                (each list corresponds to a dataset)
            data_dirs: paths to data directories
            padding: amount of padding around object crops
            crop_size: the size of the object crops. Can be either:
                - An integer specifying a single crop size for all objects
                - A list of integers specifying different crop sizes for
                  different data directories
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters.
                The keys
                should map directly to augmentation classes in albumentations. Example:
                    augs = {
                        'Rotate': {'limit': [-90, 90]},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0},
                        'RandomContrast': {'limit': 0.2}
                    }
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            max_batching_gap: the max number of frames that can be unlabelled
                before starting a new batch
            use_tight_bbox: whether to use tight bounding box (around keypoints)
                instead of the default square bounding box
            ctc_track_meta: filepaths of man_track.txt files in a list of lists
                (each list corresponds to a dataset)
            apply_mask_to_crop: whether to apply the mask to the crop
            **kwargs: Additional keyword arguments (unused but accepted for compatibility)
        """
        super().__init__(
            gt_list,
            raw_img_list,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
            ctc_track_meta,
        )

        self.raw_img_list = raw_img_list
        self.gt_list = gt_list
        self.ctc_track_meta = ctc_track_meta
        self.data_dirs = data_dirs
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode.lower()
        self.n_chunks = n_chunks
        self.seed = seed
        self.max_batching_gap = max_batching_gap
        self.use_tight_bbox = use_tight_bbox
        self.skeleton = sio.Skeleton(nodes=["centroid"])
        self.apply_mask_to_crop = apply_mask_to_crop
        if not isinstance(self.data_dirs, list):
            self.data_dirs = [self.data_dirs]

        if not isinstance(self.crop_size, list):
            # make a list so its handled consistently if multiple crops are used
            if len(self.data_dirs) > 0:  # for test mode, data_dirs is []
                self.crop_size = [self.crop_size] * len(self.data_dirs)
            else:
                self.crop_size = [self.crop_size]

        if len(self.data_dirs) > 0 and len(self.crop_size) != len(self.data_dirs):
            raise ValueError(
                f"If a list of crop sizes or data directories are given,"
                f"they must have the same length but got {len(self.crop_size)} "
                f"and {len(self.data_dirs)}"
            )

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        if augmentations and self.mode == "train":
            self.augmentations = data_utils.build_augmentations(augmentations)
        else:
            self.augmentations = None

        #
        if self.ctc_track_meta is not None:
            self.list_df_track_meta = [
                pd.read_csv(
                    gtf,
                    delimiter=" ",
                    header=None,
                    names=["track_id", "start_frame", "end_frame", "parent_id"],
                )
                for gtf in self.ctc_track_meta
            ]
        else:
            self.list_df_track_meta = None
        # frame indices for each dataset; list of lists (each list corresponds to a dataset)
        self.frame_idx = [torch.arange(len(gt_dataset)) for gt_dataset in self.gt_list]

        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks_other()

    def get_indices(self, idx: int) -> tuple:
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.

        Returns:
            the label and frame indices corresponding to a batch,
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: list[int], frame_idx: list[int]) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            a list of Frame objects containing frame metadata and Instance Objects.
            See `dreem.io.data_structures` for more info.
        """
        image_paths = self.raw_img_list[label_idx]
        gt_paths = self.gt_list[label_idx]

        # df_track_meta is currently unused but may be needed for future track metadata processing
        # if self.list_df_track_meta is not None:
        #     df_track_meta = self.list_df_track_meta[label_idx]
        # else:
        #     df_track_meta = None

        # get the correct crop size based on the video
        video_par_path = Path(image_paths[0]).parent.parent
        if len(self.data_dirs) > 0:
            crop_size = self.crop_size[0]
            for j, data_dir in enumerate(self.data_dirs):
                if Path(data_dir) == video_par_path:
                    crop_size = self.crop_size[j]
                    break
        else:
            crop_size = self.crop_size[0]

        frames = []
        max_crop_h, max_crop_w = 0, 0
        for i in frame_idx:
            instances, gt_track_ids, centroids, dict_centroids, bboxes, masks = (
                [],
                [],
                [],
                {},
                [],
                [],
            )

            i = int(i)

            img = image_paths[i]
            gt_sec = gt_paths[i]

            img = np.array(Image.open(img))
            gt_sec = np.array(Image.open(gt_sec))

            if img.dtype == np.uint16:
                img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
                    np.uint8
                )
            # if df_track_meta is None:
            unique_instances = np.unique(gt_sec)
            # else:
            # unique_instances = df_track_meta["track_id"].unique()

            for instance in unique_instances:
                # not all instances are in the frame, and they also label the
                # background instance as zero
                if instance in gt_sec and instance != 0:
                    mask = gt_sec == instance
                    center_of_mass = measurements.center_of_mass(mask)

                    # scipy returns yx
                    x, y = center_of_mass[::-1]

                    if self.use_tight_bbox:
                        bbox = data_utils.get_tight_bbox_masks(mask)
                    else:
                        bbox = data_utils.pad_bbox(
                            data_utils.get_bbox([int(x), int(y)], crop_size),
                            padding=self.padding,
                        )
                    mask = torch.as_tensor(mask)

                    gt_track_ids.append(int(instance))
                    centroids.append([x, y])
                    dict_centroids[int(instance)] = [x, y]
                    bboxes.append(bbox)
                    masks.append(mask)

            # albumentations wants (spatial, channels), ensure correct dims
            if self.augmentations is not None:
                for transform in self.augmentations:
                    # for occlusion simulation, can remove if we don't want
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                augmented = self.augmentations(
                    image=img,
                    mask=gt_sec,  # albumentations ensures geometric transformations are synced between image and mask
                    keypoints=np.vstack(centroids),
                )
                img, aug_mask, centroids = (
                    augmented["image"],
                    augmented["mask"],
                    augmented["keypoints"],
                )
                aug_mask = torch.Tensor(aug_mask).unsqueeze(0)

            img = torch.Tensor(img).unsqueeze(0)

            for j in range(len(gt_track_ids)):
                # just formatting for compatibility with Instance class
                instance_centroid = {
                    "centroid": np.array(dict_centroids[gt_track_ids[j]])
                }
                pose = {"centroid": dict_centroids[gt_track_ids[j]]}  # more formatting
                crop_raw = data_utils.crop_bbox(img, bboxes[j])
                if self.apply_mask_to_crop:
                    if (
                        self.augmentations is not None
                    ):  # TODO: change this to a flag that the user passes in apply_mask_to_crop
                        cropped_mask = data_utils.crop_bbox(aug_mask, bboxes[j])
                        # filter for the instance of interest
                        cropped_mask[cropped_mask != gt_track_ids[j]] = 0
                    else:
                        # masks[j] is already filtered for the instance of interest
                        cropped_mask = data_utils.crop_bbox(masks[j], bboxes[j])

                    cropped_mask[cropped_mask != 0] = 1
                    # apply mask to crop
                    crop = crop_raw * cropped_mask
                else:
                    crop = crop_raw

                c, h, w = crop.shape
                if h > max_crop_h:
                    max_crop_h = h
                if w > max_crop_w:
                    max_crop_w = w

                instances.append(
                    Instance(
                        gt_track_id=gt_track_ids[j],
                        pred_track_id=-1,
                        centroid=instance_centroid,
                        skeleton=self.skeleton,
                        point_scores=np.array([1.0]),
                        instance_score=np.array([1.0]),
                        pose=pose,
                        bbox=bboxes[j],
                        crop=crop,
                        mask=masks[j],
                    )
                )

            if self.mode == "train":
                np.random.shuffle(instances)

            frames.append(
                Frame(
                    video_id=label_idx,
                    frame_id=i,
                    vid_file=Path(image_paths[0]).parent.name,
                    img_shape=img.shape,
                    instances=instances,
                )
            )

        # pad bbox to max size
        if self.use_tight_bbox:
            # bound the max crop size to the user defined crop size
            max_crop_h = crop_size if max_crop_h == 0 else min(max_crop_h, crop_size)
            max_crop_w = crop_size if max_crop_w == 0 else min(max_crop_w, crop_size)
            # gather all the crops
            for frame in frames:
                for instance in frame.instances:
                    data_utils.pad_variable_size_crops(
                        instance, (max_crop_h, max_crop_w)
                    )

        return frames
