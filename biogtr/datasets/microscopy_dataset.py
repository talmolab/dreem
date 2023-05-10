"""Module containing microscopy dataset."""

from PIL import Image
from biogtr.datasets import data_utils
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf
import numpy as np
import torch


class MicroscopyDataset(Dataset):
    """Dataset for loading Microscopy Data."""

    def __init__(
        self,
        videos: list[str],
        tracks: list[str],
        source: str,
        padding: int = 5,
        crop_size: int = 20,
        chunk: bool = False,
        clip_length: int = 10,
        mode: str = "Train",
        augs: dict = None,
    ):
        """Initialize MicroscopyDataset.

        Args:
            videos: paths to raw microscopy videos
            tracks: paths to trackmate gt labels (either .xml or .csv)
            source: file format of gt labels based on label generator
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
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
        self.videos = videos
        self.tracks = tracks
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode

        self.augs = data_utils.build_augmentations(augs) if augs else None

        if source.lower() == "trackmate":
            self.parse = data_utils.parse_trackmate
        elif source.lower() == "icy":
            self.parse = data_utils.parse_ICY
        elif source.lower() == "isbi":
            self.parse = data_utils.parse_ISBI
        else:
            raise ValueError(
                f"{source} is unsupported! Must be one of [trackmate, icy, isbi]"
            )

        self.labels = [
            self.parse(self.tracks[video_idx])
            for video_idx in torch.arange(len(self.tracks))
        ]

        self.frame_idx = [
            torch.arange(Image.open(video).n_frames)
            if type(video) == str
            else torch.arange(len(video))
            for video in self.videos
        ]

        if self.chunk:
            self.chunks = [
                [i * self.clip_length for i in range(len(video) // self.clip_length)]
                for video in self.frame_idx
            ]

            self.chunked_frame_idx, self.label_idx = [], []
            for i, (split, frame_idx) in enumerate(zip(self.chunks, self.frame_idx)):
                frame_idx_split = torch.split(frame_idx, self.clip_length)
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])
        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = [i for i in range(len(self.videos))]

    def __len__(self):
        """Get length of dataset.

        Returns:
            the length of the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch):
        """Collate function used to overwrite dataloader batching function.

        Args:
            batch: the chunk of frames to be returned

        Returns:
            the batch
        """
        return batch

    def __getitem__(self, idx):
        """Get an element of the dataset.

        Returns:
            a list of dicts where each dict corresponds a frame in the chunk
            and each value is a `torch.Tensor`.

            Dict Elements:
                {
                    "video_id": The video being passed through the transformer,
                    "img_shape": the shape of each frame,
                    "frame_id": the specific frame in the entire video being used,
                    "num_detected": The number of objects in the frame,
                    "gt_track_ids": The ground truth labels,
                    "bboxes": The bounding boxes of each object,
                    "crops": The raw pixel crops,
                    "features": The feature vectors for each crop outputed by the
                        CNN encoder,
                    "pred_track_ids": The predicted trajectory labels from the
                        tracker,
                    "asso_output": the association matrix preprocessing,
                    "matches": the true positives from the model,
                    "traj_score": the association matrix post processing,
                }
        """
        label_idx = self.label_idx[idx]
        frame_idx = self.chunked_frame_idx[idx]
        labels = self.labels[label_idx]
        labels = labels.dropna(how="all")

        video = data_utils.LazyTiffStack(self.videos[label_idx])

        instances = []

        for i in frame_idx:
            gt_track_ids, centroids, bboxes, crops = [], [], [], []

            img = video.get_section(i)

            lf = labels[labels["FRAME"].astype(int) == i.item()]

            for instance in sorted(lf["TRACK_ID"].unique()):
                gt_track_ids.append(int(instance))

                x = lf[lf["TRACK_ID"] == instance]["POSITION_X"].iloc[0]
                y = lf[lf["TRACK_ID"] == instance]["POSITION_Y"].iloc[0]
                centroids.append(torch.tensor([x, y]).to(torch.float32))

            # albumentations wants (spatial, channels), ensure correct dims
            if self.augs is not None:
                augmented = self.augs(image=img, keypoints=torch.vstack(centroids))
                img, centroids = augmented["image"], augmented["keypoints"]

            for c in centroids:
                bbox = data_utils.pad_bbox(
                    data_utils.get_bbox([int(c[0]), int(c[1])], self.crop_size),
                    padding=self.padding,
                )
                bboxes.append(bbox)

            img = torch.Tensor(img)

            # torch wants (channels, spatial) - ensure correct dims
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = img.T  # todo: check for edge cases

            for bbox in bboxes:
                crop = data_utils.crop_bbox(img, bbox)
                crops.append(crop)

            instances.append(
                {
                    "video_id": torch.tensor([label_idx]),
                    "img_shape": torch.tensor([img.shape]),
                    "frame_id": torch.tensor([i]),
                    "num_detected": torch.tensor([len(bboxes)]),
                    "gt_track_ids": torch.tensor(gt_track_ids).type(torch.int64),
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
