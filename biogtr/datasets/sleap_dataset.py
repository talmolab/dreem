import torch
import imageio
import sleap_io as sio
import data_utils
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf


class SleapDataset(Dataset):
    def __init__(
        self,
        slp_files,
        padding=5,
        crop_size=128,
        anchor_names=["thorax", "head"],
        chunk=True,
        clip_length=500,
        crop_type="centroid",
        mode="train",
        tfm=None,
        tfm_cfg=None,
    ):
        """
        Dataset for loading tracking annotations stored in .slp files
        Args:
            slp_files: a list of .slp files storing tracking annotations and vid file path
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            anchor_names: names of nodes in skeleton to be used as centroid for cropping ordered by priority
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            crop_type: `centroid` or `pose` - determines whether to crop around a centroid or around a pose
            mode: `train` or `val`. Determines whether this dataset is used for training or validation.
            Currently doesn't affect dataset logic
            tfm: The augmentation function from albumentations
            tfm_cfg: The config for the augmentations
        """
        self.slp_files = slp_files
        self.padding = padding
        self.crop_size = crop_size
        self.anchor_names = anchor_names
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_type = crop_type
        self.mode = mode
        self.tfm = tfm
        self.tfm_cfg = tfm_cfg

        assert self.crop_type in ["centroid", "pose"], "Invalid crop type!"

        self.labels = [sio.load_slp(slp_file) for slp_file in self.slp_files]

        # for label in self.labels:
        # label.remove_empty_instances(keep_empty_frames=False)

        self.frame_idx = [torch.arange(len(label)) for label in self.labels]

        if self.chunk:
            self.chunks = [
                [i * self.clip_length for i in range(len(label) // self.clip_length)]
                for label in self.labels
            ]

            self.chunked_frame_idx, self.label_idx = [], []
            for i, (split, frame_idx) in enumerate(zip(self.chunks, self.frame_idx)):
                frame_idx_split = torch.split(frame_idx, split)[1:]
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])
        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = [i for i in range(len(self.labels))]

    def __len__(self) -> int:
        """
        Get the size of the dataset
        Returns the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch):
        return batch

    def __getitem__(self, idx) -> list[dict]:
        """
        Get an element of the dataset
        Returns a list of dicts where each dict corresponds a frame in the chunk and each value is a `torch.Tensor`
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
            Args:
                idx: the index of the batch. Note this is not the index of the video or the frame.
        """

        label_idx = self.label_idx[idx]
        frame_idx = self.chunked_frame_idx[idx]

        video = self.labels[label_idx]

        anchors = [
            video.skeletons[0].node_names.index(anchor_name)
            for anchor_name in self.anchor_names
        ]  # get the nodes from the skeleton

        video_name = video.videos[0].filename.split("/")[1]

        vid_reader = imageio.get_reader(video_name, "ffmpeg")

        instances = []
        for i in frame_idx:
            gt_track_ids, poses, bboxes, crops = [], [], [], []

            i = int(i)

            lf = video[i]
            lf_img = vid_reader.get_data(i)

            img = tvf.to_tensor(lf_img)

            _, h, w = img.shape

            for instance in lf:
                # gt_track_ids
                gt_track_ids.append(video.tracks.index(instance.track))

                # poses
                poses.append(torch.Tensor(instance.numpy()).astype("float32"))

                # bboxes
                if self.crop_type == "centroid":
                    bbox = data_utils.pad_bbox(
                        data_utils.centroid_bbox(instance, anchors, self.crop_size),
                        padding=self.padding,
                    )
                elif self.crop_type == "pose":
                    bbox = data_utils.pose_bbox(instance, self.padding, (w, h))

                bboxes.append(bbox)

                # crops
                if self.crop_type == "centroid":
                    crop = data_utils.crop_bbox(img, bbox)
                elif self.crop_type == "pose":
                    crop = data_utils.resize_and_pad(
                        data_utils.crop_bbox(img, bbox), self.crop_size
                    )

                crops.append(crop)

            instances.append(
                {
                    "video_id": torch.from_numpy(torch.Tensor([label_idx])),
                    "img_shape": torch.Tensor([img.shape]),
                    "frame_id": torch.Tensor([i]),
                    "num_detected": torch.Tensor([len(bboxes)]),
                    "gt_track_ids": torch.Tensor(gt_track_ids).type(torch.int64),
                    "bboxes": torch.Tensor(bboxes),
                    "crops": torch.stack(crops),
                    "features": torch.Tensor([]),
                    "pred_track_ids": torch.Tensor([-1 for _ in range(len(bboxes))]),
                    "asso_output": torch.Tensor([]),
                    "matches": torch.Tensor([]),
                    "traj_score": torch.Tensor([]),
                }
            )

        return instances
