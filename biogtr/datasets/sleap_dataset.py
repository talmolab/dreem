"""Module containing logic for loading sleap datasets."""
import albumentations as A
import torch
import imageio
import numpy as np
import sleap_io as sio
import random
import warnings
from biogtr.datasets.lazy_loaders import LazyH5
from biogtr.datasets.compute_features import (
    compute_optical_flow,
    make_edge_masks,
    make_grid_vectors,
)
from biogtr.data_structures import Frame, Instance
from biogtr.datasets import data_utils
from biogtr.datasets.base_dataset import BaseDataset
from torchvision.transforms import functional as tvf
from lsd.train import local_shape_descriptor as lsds
from typing import List, Union, Iterable


class SleapDataset(BaseDataset):
    """Dataset for loading animal behavior data from sleap."""

    def __init__(
        self,
        slp_files: list[str],
        video_files: list[str],
        feature_files: list[str] = [],
        features: Iterable = ("vis",),
        padding: int = 5,
        crop_size: int = 128,
        anchor: str = "",
        sigma: int = 30,
        chunk: bool = True,
        clip_length: int = 500,
        mode: str = "train",
        augmentations: dict = None,
        n_chunks: Union[int, float] = 1.0,
        seed: int = None,
        verbose: bool = False,
    ):
        """Initialize SleapDataset.

        Args:
            slp_files: a list of .slp files storing tracking annotations
            video_files: a list of paths to video files
            feature_files: a list of .h5 files containing precomputed features. Must have "masks" or "flows" keys.
            features: an iterable indicating which features to compute must be a set of {'visual', 'lsd', 'flow'}
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            anchor: the name of the anchor keypoint to be used as centroid for cropping.
            If unavailable then crop around the midpoint between all visible anchors.
            sigma: The distance away from the edge for which to threshold mask.
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augmentations = {
                        'Rotate': {'limit': [-90, 90], 'p': 0.5},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0, 'p': 0.2},
                        'RandomContrast': {'limit': 0.2, 'p': 0.6}
                    }
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            verbose: boolean representing whether to print
        """
        super().__init__(
            files=slp_files + video_files,
            features=features,
            padding=padding,
            crop_size=crop_size,
            chunk=chunk,
            clip_length=clip_length,
            mode=mode,
            augmentations=augmentations,
            n_chunks=n_chunks,
            seed=seed,
            verbose=verbose,
        )

        self.slp_files = slp_files
        self.video_files = video_files
        self.feature_files = feature_files

        self.padding = padding
        self.crop_size = crop_size
        self.anchor = anchor.lower()
        self.sigma = sigma

        self.chunk = chunk
        self.clip_length = clip_length

        self.n_chunks = n_chunks
        self.seed = seed

        self.mode = mode
        self.verbose = verbose

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.augmentations = (
            data_utils.build_augmentations(augmentations) if augmentations else None
        )

        self.features = [LazyH5(feature_file) for feature_file in self.feature_files]

        for feat in self.compute_feats.keys():
            if not all([feat in feature_file for feature_file in self.features]):
                self.compute_feats[feat] = True

        self.labels = [sio.load_slp(slp_file) for slp_file in self.slp_files]

        self.frame_idx = [torch.arange(len(labels)) for labels in self.labels]

        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks()
        self.profiler.time("Initialization")

    def get_indices(self, idx):
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: List[int], frame_idx: List[int]) -> list[dict]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            A list of dicts where each dict corresponds a frame in the chunk and each value is a `torch.Tensor`
            Dict Elements:
            {
                        "video_id": The video being passed through the transformer,
                        "img_shape": the shape of each frame,
                        "frame_id": the specific frame in the entire video being used,
                        "num_detected": The number of objects in the frame,
                        "gt_track_ids": The ground truth labels,
                        "bboxes": The bounding boxes of each object,
                        "crops": The raw pixel crops,
                        "features": The feature vectors for each crop outputed by the CNN encoder,
                        "pred_track_ids": The predicted trajectory labels from the tracker,
                        "asso_output": the association matrix preprocessing,
                        "matches": the true positives from the model,
                        "traj_score": the association matrix post processing,
                }

        """
        video = self.labels[label_idx]

        video_name = self.video_files[label_idx]

        vid_reader = imageio.get_reader(video_name, "ffmpeg")

        sources, sinks = [], []
        for edge in video.skeleton.edges:
            sources.append(edge.source.name)
            sinks.append(edge.destination.name)
        # print(list(zip(sources, sinks)))

        if len(self.features) > 0:
            features = self.features[label_idx]
        else:
            features = None

        frames = []
        for i, frame_ind in enumerate(frame_idx):
            instances, gt_track_ids, shown_poses = [], [], []

            frame_ind = int(frame_ind)

            lf = video[frame_ind]

            # Read frame from video
            try:
                img = vid_reader.get_data(frame_ind)
                self.profiler.time(f"Reading frame {frame_ind}")
                img_shape = img.shape
                if self.compute_feats["flows"] and self.return_feats["flows"]:
                    if frame_ind > 0:
                        prev_img = vid_reader.get_data(frame_ind - 1)
                        self.profiler.time(f"Reading frame {frame_ind - 1}")
                    else:
                        prev_img = torch.zeros_like(torch.tensor(img))
                else:
                    prev_img = np.empty(img_shape)

            except IndexError as e:
                print(f"Could not read frame {frame_ind} from {video_name} due to {e}")
                continue

            # Collate poses and masks from instances
            for instance in lf:
                gt_track_id = video.tracks.index(
                    instance.track
                )  # get instance gt label
                self.profiler.time(f"Getting gt_track_id")
                gt_track_ids.append(gt_track_id)

                ##################
                # Get poses
                shown_poses.append(
                    dict(
                        zip(
                            [n.name for n in instance.skeleton.nodes],
                            [[p.x, p.y] for p in instance.points.values()],
                        )
                    )
                )

                shown_poses = [
                    {
                        key.lower(): val
                        for key, val in instance.items()
                        if not np.isnan(val).any()
                    }
                    for instance in shown_poses
                ]
                self.profiler.time("Getting poses")
                #
                ####################

            ##################################
            # Augment images, keypoints, masks, and flows (?)
            if self.augmentations is not None:
                for transform in self.augmentations:
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                if shown_poses:
                    keypoints = np.vstack([list(s.values()) for s in shown_poses])

                else:
                    keypoints = np.array([])

                if self.return_feats["vis"]:
                    aug_img = img
                else:
                    aug_img = np.empty(img_shape)

                augmented = self.augmentations(
                    image=aug_img, image2=prev_img, keypoints=keypoints
                )
                self.profiler.time("Augmenting data")
                img, prev_img, aug_poses = (
                    augmented["image"],
                    augmented["image2"],
                    augmented["keypoints"],
                )

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
            #
            ############################

            self.profiler.time("Calculating optical flow")
            # Get crops, lsd, features and combine into Instance object
            for i in range(len(gt_track_ids)):
                pose = shown_poses[i]

                ######################
                # Compute Bboxes
                """Check for anchor"""
                if self.anchor in pose:
                    anchor = self.anchor
                else:
                    if self.verbose:
                        warnings.warn(
                            f"{self.anchor} not in {[key for key in pose.keys()]}! Defaulting to midpoint"
                        )
                    anchor = "midpoint"

                if anchor != "midpoint":
                    centroid = pose[anchor]

                    if not np.isnan(centroid).any():
                        bbox = data_utils.pad_bbox(
                            data_utils.get_bbox(centroid, self.crop_size),
                            padding=self.padding,
                        )

                    else:
                        # print(f'{self.anchor} contains NaN: {centroid}. Using midpoint')
                        bbox = data_utils.pad_bbox(
                            data_utils.pose_bbox(
                                np.array(list(pose.values())), self.crop_size
                            ),
                            padding=self.padding,
                        )
                else:
                    # print(f'{self.anchor} not an available option amongst {pose.keys()}. Using midpoint')
                    bbox = data_utils.pad_bbox(
                        data_utils.pose_bbox(
                            np.array(list(pose.values())), self.crop_size
                        ),
                        padding=self.padding,
                    )
                self.profiler.time("Computing bbox coords")
                #
                ############################

                ############################
                # Get crops

                if self.return_feats["lsds"] and not self.compute_feats["masks"]:
                    mask = features.get_section("masks", frame_ind)
                else:
                    mask = np.zeros_like(img)

                crop, pose_coords, mask = data_utils.crop_bbox(
                    img, bbox, np.array(list(pose.values())), mask
                )

                pose = {
                    key.lower(): pose_coords[i] for i, key in enumerate(pose.keys())
                }
                self.profiler.time("Cropping image")
                #
                ############################

                ############################
                # Compute LSD
                if self.return_feats["lsds"]:
                    ##########################
                    # Compute Raster Mask
                    if self.compute_feats["masks"]:
                        source_coords, sink_coords = [], []
                        for source_node, sink_node in zip(sources, sinks):
                            if (
                                source_node.lower() in pose.keys()
                                and sink_node.lower() in pose.keys()
                            ):
                                source_coords.append(pose[source_node.lower()])
                                sink_coords.append(pose[sink_node.lower()])

                        source_coords = torch.tensor(source_coords)
                        sink_coords = torch.tensor(sink_coords)

                        # print(pose)
                        # print(source_coords, sink_coords)
                        xv, yv = make_grid_vectors(crop.shape[0], crop.shape[1])
                        mask = make_edge_masks(
                            xv,
                            yv,
                            source_coords,
                            sink_coords,
                            self.sigma,
                            gt_track_ids[i],
                        )
                        self.profiler.time("Computing Masks")

                    lsd = lsds.get_local_shape_descriptors(
                        mask, sigma=(5, 5), voxel_size=(1, 1), downsample=2
                    )
                    #
                    ############################
                else:
                    lsd = []

                ############################

                if self.compute_feats["flows"] and self.return_feats["flows"]:
                    if frame_ind > 0:
                        prev_crop, _, _ = data_utils.crop_bbox(prev_img, bbox)
                        flow = compute_optical_flow(
                            prev_crop, crop, downsample_factor=1.0
                        )
                    else:
                        flow = torch.zeros(
                            (
                                2,
                                self.crop_size + 2 * self.padding,
                                self.crop_size + 2 * self.padding,
                            )
                        )
                elif self.return_feats["flows"] and not self.compute_feats["flows"]:
                    flow, _, _ = features.get_section("flows", frame_ind)
                    flow, _, _ = data_utils.crop_bbox(flow, bbox)
                else:
                    flow = []

                self.profiler.time("Getting instance flows")

                instance = Instance(
                    gt_track_id=gt_track_ids[i],
                    pred_track_id=-1,
                    crop=crop,
                    bbox=bbox,
                    pose=pose,
                    mask=mask,
                    lsd=lsd,
                    flow=flow,
                )

                self.profiler.time("Creating instance")
                instances.append(instance)

            frame = Frame(
                video_id=label_idx,
                frame_id=frame_ind,
                img_shape=img_shape,
                instances=instances,
            )
            self.profiler.time("Creating frame")
            frames.append(frame)

        return frames
