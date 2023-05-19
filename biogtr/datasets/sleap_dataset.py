"""Module containing logic for loading sleap datasets."""
import torch
import imageio
import numpy as np
import sleap_io as sio
import random
from biogtr.datasets import data_utils
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf


class SleapDataset(Dataset):
    """Dataset for loading animal behavior data from sleap."""

    def __init__(
        self,
        slp_files: list[str],
        video_files: list[str],
        padding: int = 5,
        crop_size: int = 128,
        chunk: bool = True,
        clip_length: int = 500,
        crop_type: str = "centroid",
        mode: str = "train",
        augs: dict = None,
    ):
        """Initialize Sleap Dataset.

        Args:
            slp_files: a list of .slp files storing tracking annotations
            video_files: a list of paths to video files
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            crop_type: `centroid` or `pose` - determines whether to crop around
                a centroid or around a pose
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augs: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augs = {
                        'Rotate': {'limit': [-90, 90]},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0},
                        'RandomContrast': {'limit': 0.2}
                    }
        """

        self.slp_files = slp_files
        self.video_files = video_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_type = crop_type
        self.mode = mode

        self.augs = data_utils.build_augmentations(augs) if augs else None

        assert self.crop_type in ["centroid", "pose"], "Invalid crop type!"

        self.labels = [sio.load_slp(slp_file) for slp_file in self.slp_files]

        self.anchor_names = [
            data_utils.sorted_anchors(labels) for labels in self.labels
        ]

        self.frame_idx = [torch.arange(len(label)) for label in self.labels]

        if self.chunk:
            self.chunks = [
                [i * self.clip_length for i in range(len(label) // self.clip_length)]
                for label in self.labels
            ]

            self.chunked_frame_idx, self.label_idx = [], []
            for i, (split, frame_idx) in enumerate(zip(self.chunks, self.frame_idx)):
                frame_idx_split = torch.split(frame_idx, self.clip_length)
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])
        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = [i for i in range(len(self.labels))]

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch):
        """Collate function used to overwrite dataloader batching function.

        Args:
            batch: the chunk of frames to be returned

        Returns:
            The batch
        """
        return batch

    def __getitem__(self, idx) -> list[dict]:
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video or the frame.

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
        label_idx = self.label_idx[idx]
        frame_idx = self.chunked_frame_idx[idx]

        video = self.labels[label_idx]

        anchors = [
            video.skeletons[0].node_names.index(anchor_name)
            for anchor_name in self.anchor_names[label_idx]
        ]  # get the nodes from the skeleton

        video_name = self.video_files[label_idx]

        vid_reader = imageio.get_reader(video_name, "ffmpeg")

        instances = []

        for i in frame_idx:
            gt_track_ids, bboxes, crops, poses = [], [], [], []

            i = int(i)

            lf = video[i]
            lf_img = vid_reader.get_data(i)

            # albumentations requires np array, crop wants tensors. convert to
            # tensor after augmenting

            for instance in lf:
                # gt_track_ids
                gt_track_ids.append(video.tracks.index(instance.track))

                # # poses
                # might be necessary later so commenting out for now
                # poses.append(torch.Tensor(instance.numpy()).astype("float32"))

                # should we interpolate nan values instead?
                for anchor in anchors:
                    cx, cy = instance[anchor].x, instance[anchor].y
                    poses.append([cx, cy])
                    if not np.isnan(cx):
                        break

            # augs
            if self.augs is not None:
                augmented = self.augs(image=lf_img, keypoints=np.vstack(poses))
                img, _ = augmented["image"], augmented["keypoints"]

            img = tvf.to_tensor(img)

            _, h, w = img.shape

            # don't like this repeated loop through frames. Todo: change
            # cropping to np arrays, then finally convert to tensors.
            for instance in lf:
                if self.crop_type == "centroid":
                    bbox = data_utils.pad_bbox(
                        data_utils.centroid_bbox(instance, anchors, self.crop_size),
                        padding=self.padding,
                    )
                    crop = data_utils.crop_bbox(img, bbox)

                elif self.crop_type == "pose":
                    # this is broken, sleap-io handles points differently
                    # and torch.nanmin seems deprecated. fix pose_bbox function
                    bbox = data_utils.pose_bbox(instance, self.padding, (w, h))
                    crop = data_utils.resize_and_pad(
                        data_utils.crop_bbox(img, bbox), self.crop_size
                    )

                bboxes.append(bbox)
                crops.append(crop)

            instances.append(
                {
                    "video_id": torch.tensor([label_idx]),
                    "img_shape": torch.tensor([img.shape]),
                    "frame_id": torch.tensor([i]),
                    "num_detected": torch.tensor([len(bboxes)]),
                    "gt_track_ids": torch.tensor(gt_track_ids),
                    "bboxes": torch.stack(bboxes),
                    "crops": torch.stack(crops),
                    "features": torch.tensor([]),
                    "pred_track_ids": torch.tensor([-1 for _ in range(len(bboxes))]),
                    "asso_output": torch.tensor([]),
                    "matches": torch.tensor([]),
                    "traj_score": torch.tensor([]),
                }
            )

        return instances
