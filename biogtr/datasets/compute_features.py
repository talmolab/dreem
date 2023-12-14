"""Module containing logic for computing additional features such as optical flow and LSDs."""
import h5py
import hydra
import imageio.v3 as imageio
import dataclasses
import sleap_io as sio
import numpy as np
import cv2
import torch
import albumentations as A

from pathlib import Path
from biogtr.datasets.data_utils import expand_to_rank
from tqdm import tqdm
from cv2 import calcOpticalFlowFarneback

from omegaconf import DictConfig, OmegaConf, MISSING

defaults = [{"cfg": "default"}]


def compute_optical_flow(
    prev: np.ndarray,
    curr: np.ndarray,
    normalize: bool = True,
    downsample_factor: float = 1.0,
) -> torch.TensorType:
    """Compute dense optical flow between frames.

    Args:
        prev: The np array of shape (h, w, c) containing the previous image
        curr: The np array of shape (h, w, c) containing the current frame for which to compute optical flow
        normalize: Whether or not to normalize optical flow between 0-1 after calculation
        downsample_factor: Fraction by which to downsample images before computing optical flow. Useful for speeding up computation.
    """
    frame1 = prev
    frame2 = curr
    if frame1.shape[-1] > 3:
        frame1.transpose(1, 2, 0)
    if frame2.shape[-1] > 3:
        frame1.transpose(1, 2, 0)
    img_shape = frame1.shape
    # print(frame1.shape, frame2.shape, img_shape)

    resize_down = A.augmentations.geometric.resize.Resize(
        int(img_shape[0] * downsample_factor),
        int(img_shape[1] * downsample_factor),
    )
    resize_up = A.augmentations.geometric.resize.Resize(img_shape[0], img_shape[1])

    if downsample_factor < 1.0:
        frame1 = resize_down(image=frame1)["image"]
        frame2 = resize_down(image=frame2)["image"]

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    of = calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if downsample_factor < 1.0:
        of = resize_up(image=of)["image"]

    if normalize:
        of = (of - np.min(of)) / (np.max(of) - np.min(of) + 1e-8)

    return torch.tensor(of).permute(-1, 0, 1)


def distance_to_edge(
    points: torch.TensorType,
    edge_source: torch.TensorType,
    edge_destination: torch.TensorType,
) -> torch.TensorType:
    """Compute pairwise distance between points and undirected edges.

    Args:
        points: Tensor of dtype torch.float32 of shape (d_0, ..., d_n, 2) where the last
            axis corresponds to x- and y-coordinates. Distances will be broadcast across
            all point dimensions.
        edge_source: Tensor of dtype torch.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype torch.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the source points of each
            edge.

    Returns:
        A tensor of dtype torch.float32 of shape (d_0, ..., d_n, n_edges) where the first
        axes correspond to the initial dimensions of `points`, and the last indicates
        the distance of each point to each edge.
    """
    n_pt_dims = torch.linalg.matrix_rank(points).max() - 1

    direction_vector = edge_destination - edge_source

    edge_length = torch.maximum(
        torch.sum(torch.square(direction_vector), axis=1), torch.tensor(1)
    )

    source_relative_points = torch.unsqueeze(points, axis=-2) - expand_to_rank(
        edge_source, n_pt_dims + 2
    )

    line_projections = torch.sum(
        source_relative_points * expand_to_rank(direction_vector, n_pt_dims + 2), axis=3
    ) / expand_to_rank(
        edge_length, n_pt_dims + 1
    )  # (..., n_edges)

    line_projections = torch.clamp(line_projections, 0, 1)

    # print(line_projections.unsqueeze(-1).shape, expand_to_rank(direction_vector, n_pt_dims + 2).shape)
    distances = torch.sum(
        torch.square(
            (
                line_projections.unsqueeze(-1)
                * expand_to_rank(direction_vector, n_pt_dims + 2)
            )
            - source_relative_points
        ),
        axis=-1,
    )  # (..., n_edges)

    return distances


def get_sources_and_sinks(
    instance: sio.Instance, skeleton: sio.Skeleton
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split instance coordinates into source and sink nodes based on skeleton.

    Args:
        instance: the sleap_io instance for which to get sources and sink nodes
        skeleton: a sleap_io skeleton representing the directed graph describing
        the skeleton structure for a pose
    Returns:
        instance_sources: An (e, 2) tensor indicating the spatial position of each source node in the skeleton
        where e represents the number of edges and the second dimension is the xy coordinates.
        instance_destinations: An (e, 2) tensor indicating the spatial position of each sink node in the skeleton
        where e represents the number of edges and the second dimension is the xy coordinates.
    """
    instance_sources, instance_destinations = [], []

    for edge in skeleton.edges:
        source = edge.source.name
        destination = edge.destination.name
        instance_sources.append(instance.__getitem__(source).numpy())
        instance_destinations.append(instance.__getitem__(destination).numpy())

    instance_sources = np.stack(instance_sources)
    instance_destinations = np.stack(instance_destinations)

    if instance_sources.shape[-1] > 2:
        instance_sources = instance_sources[:, :-1]
    if instance_destinations.shape[-1] > 2:
        instance_destinations = instance_destinations[:, :-1]

    nan_indices = ~np.isnan(
        np.stack([instance_sources, instance_destinations], axis=-1)
    ).any(axis=(1, 2))

    instance_sources = instance_sources[
        nan_indices,
        :,
    ]

    instance_destinations = instance_destinations[
        nan_indices,
        :,
    ]

    return torch.tensor(instance_sources), torch.tensor(instance_destinations)


def make_grid_vectors(
    image_height: int, image_width: int, output_stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make sampling grid vectors from image dimensions.

        This is a useful function for creating the x- and y-vectors that define a sampling
        grid over an image space. These vectors can be used to generate a full meshgrid or
        for equivalent broadcasting operations.
        ​
        Args:
            image_height: Height of the image grid that will be sampled, specified as a
                scalar integer.
            image_width: width of the image grid that will be sampled, specified as a
                scalar integer.
            output_stride: Sampling step size, specified as a scalar integer. This can be
                used to specify a sampling grid that has a smaller shape than the image
                grid but with values span the same range. This can be thought of as the
                reciprocal of the output scale, i.e., it will induce subsampling when set to
                values greater than 1.
    ​
        Returns:
            Tuple of grid vectors (xv, yv). These are tensors of dtype tf.float32 with
            shapes (grid_width,) and (grid_height,) respectively.
    ​
            The grid dimensions are calculated as:
                grid_width = image_width // output_stride
                grid_height = image_height // output_stride
    """
    xv = torch.arange(0, image_width, step=output_stride)
    xv = xv.to(torch.float32)
    yv = torch.arange(0, image_height, step=output_stride)
    yv = yv.to(torch.float32)
    return xv, yv


def make_edge_masks(
    xv: torch.Tensor,
    yv: torch.Tensor,
    edge_source: torch.Tensor,
    edge_destination: torch.Tensor,
    sigma: float,
    instance_id: int,
) -> torch.Tensor:
    """Generate raster masks for a set of undirected edges.

    Args:
        xv: Sampling grid vector for x-coordinates of shape (grid_width,) and dtype
            tf.float32. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        yv: Sampling grid vector for y-coordinates of shape (grid_height,) and dtype
            tf.float32. This can be generated by
            `sleap.nn.data.utils.make_grid_vectors`.
        edge_source: Tensor of dtype tf.float32 of shape (n_edges, 2) where the last
            axis corresponds to x- and y-coordinates of the source points of each edge.
        edge_destination: Tensor of dtype tf.float32 of shape (n_edges, 2) where the
            last axis corresponds to x- and y-coordinates of the destination points of
            each edge.
        sigma: The distance threshold in pixels away from the edge to mask.
        If d < sigma then mask_xy = 1, otherwise = 0.

    Returns:
        A binary mask rasterizing the skeleton.
    """
    sampling_grid = torch.stack(torch.meshgrid(xv, yv), axis=-1)  # (height, width, 2)
    distances = distance_to_edge(
        sampling_grid, edge_source=edge_source, edge_destination=edge_destination
    )
    edge_maps = torch.where(distances < sigma, instance_id + 1, 0).numpy().max(axis=-1)
    return edge_maps.T


def compute_image_mask(
    lf: sio.LabeledFrame, sigma: float, h: int, w: int
) -> np.ndarray:
    """Compute mask for labeled frame.

    Args:
        lf: the sleap_io.LabeledFrame object for which to compute a mask.
        sigma: The distance threshold in pixels away from the edge to mask.
        h: Height of the image grid that will be sampled, specified as a
                scalar integer.
        w: width of the image grid that will be sampled, specified as a
            scalar integer.

    Returns:
        An np array of shape (h, w) containing a multiclass mask for the frame, where each class is an instance in the frame.
    """
    masks = []
    for j, instance in enumerate(lf):
        sources, sinks = get_sources_and_sinks(instance, instance.skeleton)
        xv, yv = make_grid_vectors(h, w)
        mask = make_edge_masks(
            xv,
            yv,
            sources,
            sinks,
            sigma=sigma,
            instance_id=j,
        )
        masks.append(mask)
    return np.stack(masks, axis=0).max(axis=0)


@dataclasses.dataclass
class DefaultConfig:
    """Default parameters for compute_features.py script."""

    in_files: list[str] = MISSING
    label_files: list[str] = MISSING
    compute: list[str] = dataclasses.field(default_factory=lambda: ["flows", "masks"])
    sigma: int = 30
    normalize: bool = True
    downsample_factor: float = 0.5


@dataclasses.dataclass
class Config:
    """Hydra config."""

    defaults: list = dataclasses.field(default_factory=lambda: defaults)
    cfg = MISSING


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(group="cfg", name="default", node=DefaultConfig)
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def compute_features(cfg: DictConfig):
    """Compute features and save them to an h5 file.

    Features are saved to the same path as in_files just with a .h5 extension.

    Args:
        cfg: A DictConfig containing the following parameters:
                in_files: A list of videos for which to compute features for.
                label_files: A list of corresponding .slp labels files.
                compute: A list of features to compute. Must be one of {masks, flows}.
                The distance threshold in pixels away from the edge to mask.
                normalize: Whether to normalize optical flow between 0-1 after computation
                downsample_factor: Fraction by which to downsample images before computing optical flow. Useful for speeding up computation.
    """
    cfg = cfg.cfg
    for in_file, label_file in tqdm(
        zip(cfg.in_files, cfg.label_files), desc="Video", unit="videos"
    ):
        out_file = Path(label_file).with_suffix(".hdf5")
        vid = imageio.imopen(in_file, "r")

        labels = sio.load_slp(label_file)
        with h5py.File(out_file, "w") as of:
            for feat in cfg.compute:
                feats = of.create_group(feat)
                for i, lf in tqdm(enumerate(labels), desc=feat, unit="frames"):
                    curr_frame = vid.read(index=i)
                    img_shape = curr_frame.shape

                    if feat == "masks":
                        frame_feat = compute_image_mask(
                            lf, cfg.sigma, img_shape[0], img_shape[1]
                        )
                    elif feat == "flows":
                        if i == 0:
                            frame_feat = np.zeros((2, img_shape[0], img_shape[1]))
                        else:
                            prev_frame = vid.read(index=i - 1)
                            frame_feat = compute_optical_flow(
                                prev_frame,
                                curr_frame,
                                normalize=cfg.normalize,
                                downsample_factor=cfg.downsample_factor,
                            )
                    feats.create_dataset(name=f"{i}", data=frame_feat)


if __name__ == "__main__":
    """Usage:
    `python compute_features.py "cfg.in_files=["VID1", "VID2", ...]" "cfg.label_files=["LABELS1", "LABELS2", ...]" "cfg.compute=["masks", "flows"]" cfg.sigma=30 cfg.normalize=True`
    """
    compute_features()
