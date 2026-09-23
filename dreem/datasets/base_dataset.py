"""Module containing logic for loading datasets."""

import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from dreem.datasets import data_utils
from dreem.io import Frame

logger = logging.getLogger("dreem.datasets")


class BaseDataset(Dataset):
    """Base Dataset for microscopy and sleap datasets to override."""

    def __init__(
        self,
        label_files: list[str],
        vid_files: list[str],
        padding: int,
        crop_size: Union[int, list[int]],
        chunk: bool,
        clip_length: int,
        mode: str,
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
        gt_list: str | None = None,
    ):
        """Initialize Dataset.

        Args:
            label_files: a list of paths to label files. Should at least contain
                detections for inference, detections + tracks for training.
            vid_files: list of paths to video files.
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters.
                See subclasses for details.
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            gt_list: An optional path to .txt file containing ground truth for
                cell tracking challenge datasets.
        """
        self.vid_files = vid_files
        self.label_files = label_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.mode = mode
        self.n_chunks = n_chunks
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        if augmentations and self.mode == "train":
            self.instance_dropout = augmentations.pop(
                "InstanceDropout", {"p": 0.0, "n": 0}
            )
            self.node_dropout = data_utils.NodeDropout(
                **augmentations.pop("NodeDropout", {"p": 0.0, "n": 0})
            )
            self.augmentations = data_utils.build_augmentations(augmentations)
        else:
            self.instance_dropout = {"p": 0.0, "n": 0}
            self.node_dropout = data_utils.NodeDropout(p=0.0, n=0)
            self.augmentations = None

        # Initialize in subclasses
        self.frame_idx = None
        self.labels = None
        self.gt_list = None

    def process_segments(
        self, i: int, segments_to_stitch: list[torch.Tensor], clip_length: int
    ) -> None:
        """Process segments to stitch.

        Modifies state variables chunked_frame_idx and label_idx.

        Args:
            segments_to_stitch: list of segments to stitch
            i: index of the video
            clip_length: the number of frames in each chunk
        Returns: None
        """
        stitched_segment = torch.cat(segments_to_stitch)
        if getattr(self, "burst_aug", False) and self.chunk:
            frame_idx_split = self._burst_chunks(stitched_segment, clip_length)
        else:
            frame_idx_split = list(torch.split(stitched_segment, clip_length))
        self.chunked_frame_idx.extend(frame_idx_split)
        self.label_idx.extend(len(frame_idx_split) * [i])

    def _burst_chunks(
        self, seg: torch.Tensor, clip_length: int
    ) -> list[torch.Tensor]:
        """Chunks made of several short bursts of CONSECUTIVE frames, far apart.

        Why not evenly-strided chunks (the `gap_aug` design): a chunk strided over
        a 900-frame span at clip_length 32 has a uniform stride of 28, so its
        *smallest* pairwise gap is 28. It contains no short-range pairs at all.
        Mixing such chunks with contiguous ones teaches the two regimes in
        separate windows, and the model is never made to tell them apart.

        A burst chunk holds both in ONE window. With clip_length 32 and 4 bursts
        of 8, its pairwise gaps are 1-7 within a burst (position is nearly
        decisive) and up to `burst_max_span` across bursts (appearance must
        carry it). The association loss scores every pair in the window
        (`AssoLoss` flattens all instances across all frames), so a single
        forward pass supervises both regimes and the contrast between them.

        It is also the shape of a real re-ID event -- animal visible, gone,
        visible again -- so natural occlusions become a special case of the
        training distribution rather than a scarce resource to sample around.

        ⚠ Needs a temporal embedding that can express the gaps. `mode: learned`
        clamps into a lookup indexed by window position, so every across-burst
        gap collapses onto one row and this degenerates into noise; use
        `mode: fixed` (RFF). A single `sigma` resolves about one decade, so
        sigma ~ burst_len/3 separates within-burst gaps and floors across-burst
        ones -- enough for "near vs far", not enough to grade the far end.

        Frames are drawn from the labeled `seg`, so every selected frame is
        labeled, and `pos` advances by `clip_length` (not by the span) so chunk
        starts stay dense and the clip is covered about as often as with plain
        contiguous chunking.
        """
        n_bursts = max(1, int(getattr(self, "n_bursts", 4)))
        max_span_cfg = int(getattr(self, "burst_max_span", 900))

        # burst lengths as even as possible; the remainder goes to the first ones
        base, rem = divmod(clip_length, n_bursts)
        if base == 0:  # more bursts than frames: degenerate to 1 frame each
            base, rem, n_bursts = 1, 0, clip_length
        lens = [base + (1 if k < rem else 0) for k in range(n_bursts)]

        chunks: list[torch.Tensor] = []
        pos, n = 0, len(seg)
        while pos < n:
            remaining = n - pos
            if remaining < clip_length:
                chunks.append(seg[pos:n])  # short consecutive tail
                break
            max_span = min(max_span_cfg, remaining)
            if max_span <= clip_length:
                chunks.append(seg[pos : pos + clip_length])  # no room to spread
                pos += clip_length
                continue

            span = int(np.exp(np.random.uniform(np.log(clip_length), np.log(max_span))))
            span = max(clip_length, min(span, max_span))

            # Place the bursts without overlap: draw sorted cut points in the
            # free space, then push each burst past the ones before it. This
            # guarantees start_{k+1} - start_k >= lens[k] and the last burst
            # still ends inside the span.
            free = span - clip_length
            cuts = np.sort(np.random.randint(0, free + 1, size=n_bursts))
            offsets, run = [], 0
            for k in range(n_bursts):
                offsets.append(int(cuts[k]) + run)
                run += lens[k]

            sel = torch.cat(
                [torch.arange(o, o + L) for o, L in zip(offsets, lens)]
            )
            window = seg[pos : pos + span]
            chunks.append(window[sel])
            pos += clip_length
        return chunks

    def create_chunks_slp(self) -> None:
        """Get indexing for data.

        Creates both indexes for selecting dataset (label_idx) and frame in
        dataset (chunked_frame_idx). If chunking is false, we index directly
        using the frame ids. Setting chunking to true creates a list of lists
        containing chunk frames for indexing. This is useful for computational
        efficiency and data shuffling. To be called by subclass __init__()
        """
        self.chunked_frame_idx, self.label_idx = [], []
        # go through each slp file and create chunks that respect max_batching_gap
        for i, slp_file in enumerate(self.label_files):
            annotated_segments = self.annotated_segments[slp_file]
            segments_to_stitch = []
            prev_end = annotated_segments[0][1]  # end of first segment
            for start, end in annotated_segments:
                # check if the start of current segment is within
                # batching_max_gap of end of previous
                if (
                    (int(start) - int(prev_end) < self.max_batching_gap)
                    or not self.chunk
                ):  # also takes care of first segment as start < prev_end
                    segments_to_stitch.append(torch.arange(start, end + 1))
                    prev_end = end
                else:
                    # stitch previous set of segments before creating a new chunk
                    self.process_segments(i, segments_to_stitch, self.clip_length)
                    # reset segments_to_stitch as we are starting a new chunk
                    segments_to_stitch = [torch.arange(start, end + 1)]
                    prev_end = end

            if not self.chunk:
                self.process_segments(
                    i, segments_to_stitch, self.labels[i].video.shape[0]
                )
            else:
                # add last chunk after the loop
                if segments_to_stitch:
                    self.process_segments(i, segments_to_stitch, self.clip_length)

        if self.n_chunks > 0 and self.n_chunks <= 1.0:
            n_chunks = int(self.n_chunks * len(self.chunked_frame_idx))

        elif self.n_chunks <= len(self.chunked_frame_idx):
            n_chunks = int(self.n_chunks)

        else:
            n_chunks = len(self.chunked_frame_idx)

        if n_chunks > 0 and n_chunks < len(self.chunked_frame_idx):
            sample_idx = np.random.choice(
                np.arange(len(self.chunked_frame_idx)), n_chunks, replace=False
            )

            self.chunked_frame_idx = [self.chunked_frame_idx[i] for i in sample_idx]

            self.label_idx = [self.label_idx[i] for i in sample_idx]

        # workaround for empty batch bug (needs to be changed).
        # Check for batch with with only 1/10 size of clip length.
        # Arbitrary thresholds
        remove_idx = []
        for i, frame_chunk in enumerate(self.chunked_frame_idx):
            if (
                len(frame_chunk) <= min(int(self.clip_length / 10), 5)
                # and frame_chunk[-1] % self.clip_length == 0
            ):
                logger.warning(
                    f"Warning: Batch containing frames {frame_chunk} from video "
                    f"{self.vid_files[self.label_idx[i]]} has {len(frame_chunk)} frames. "
                    f"Removing to avoid empty batch possibility with failed frame loading"
                )
                remove_idx.append(i)
        if len(remove_idx) > 0:
            for i in sorted(remove_idx, reverse=True):
                self.chunked_frame_idx.pop(i)
                self.label_idx.pop(i)

    def create_chunks_other(self) -> None:
        """Legacy chunking logic. Does not support unannotated segments.

        Creates both indexes for selecting dataset (label_idx) and frame in
        dataset (chunked_frame_idx). If chunking is false, we index directly
        using the frame ids. Setting chunking to true creates a list of lists
        containing chunk frames for indexing. This is useful for computational
        efficiency and data shuffling. To be called by subclass __init__()
        """
        if self.chunk:
            self.chunked_frame_idx, self.label_idx = [], []
            for i, frame_idx in enumerate(self.frame_idx):
                frame_idx_split = torch.split(frame_idx, self.clip_length)
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])

            if self.n_chunks > 0 and self.n_chunks <= 1.0:
                n_chunks = int(self.n_chunks * len(self.chunked_frame_idx))

            elif self.n_chunks <= len(self.chunked_frame_idx):
                n_chunks = int(self.n_chunks)

            else:
                n_chunks = len(self.chunked_frame_idx)

            if n_chunks > 0 and n_chunks < len(self.chunked_frame_idx):
                sample_idx = np.random.choice(
                    np.arange(len(self.chunked_frame_idx)), n_chunks, replace=False
                )

                self.chunked_frame_idx = [self.chunked_frame_idx[i] for i in sample_idx]

                self.label_idx = [self.label_idx[i] for i in sample_idx]

            # workaround for empty batch bug (needs to be changed).
            # Check for batch with with only 1/10 size of clip length.
            # Arbitrary thresholds
            remove_idx = []
            for i, frame_chunk in enumerate(self.chunked_frame_idx):
                if (
                    len(frame_chunk) <= min(int(self.clip_length / 10), 5)
                    # and frame_chunk[-1] % self.clip_length == 0
                ):
                    logger.warning(
                        f"Warning: Batch containing frames {frame_chunk} from video {self.vid_files[self.label_idx[i]]} has {len(frame_chunk)} frames. Removing to avoid empty batch possibility with failed frame loading"
                    )
                    remove_idx.append(i)
            if len(remove_idx) > 0:
                for i in sorted(remove_idx, reverse=True):
                    self.chunked_frame_idx.pop(i)
                    self.label_idx.pop(i)

        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = list(range(len(self.frame_idx)))

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch: list[Frame]) -> list[Frame]:
        """Collate function used to overwrite dataloader batching function.

        Args:
            batch: the chunk of frames to be returned

        Returns:
            The batch
        """
        return batch

    def __getitem__(self, idx: int) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video
                or the frame.

        Returns:
            A list of `Frame`s in the chunk containing the metadata + instance features.
        """
        label_idx, frame_idx = self.get_indices(idx)

        return self.get_instances(label_idx, frame_idx)

    def get_indices(self, idx: int):
        """Retrieve label and frame indices given batch index.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            idx: the index of the batch.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_instances(self, label_idx: list[int], frame_idx: list[int]):
        """Build chunk of frames.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            label_idx: The index of the labels.
            frame_idx: The index of the frames.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")
