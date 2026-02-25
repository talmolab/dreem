"""Conversion utilities for importing tracking data from external formats to .slp files."""

import numpy as np
import pandas as pd
import sleap_io as sio
import imageio
from pathlib import Path
from tifffile import imread
from tqdm import tqdm


def make_labels(video_path: str, trajectories_path: str) -> sio.Labels:
    """Convert a TrackMate trajectories file and video to a SLEAP Labels object.

    Args:
        video_path: Path to the video file (will look for .mp4 version).
        trajectories_path: Path to the TrackMate CSV or XLSX trajectories file.

    Returns:
        A SLEAP Labels object containing the converted tracking data.
    """
    vid = sio.Video.from_filename(str(Path(video_path).with_suffix(".mp4")))

    if "csv" in trajectories_path:
        traj = pd.read_csv(trajectories_path, encoding="ISO-8859-1")
    elif "xlsx" in trajectories_path:
        traj = pd.read_excel(trajectories_path)
    else:
        raise ValueError(
            f"Must be either csv or xlsx file. Got {Path(trajectories_path).suffix}!"
        )

    traj = traj.apply(pd.to_numeric, errors="coerce", downcast="integer")
    traj = traj.drop(range(0, 3), axis=0)

    posx_key = "POSITION_X"
    posy_key = "POSITION_Y"
    frame_key = "FRAME"
    track_key = "TRACK_ID"

    mapper = {
        "X": posx_key,
        "Y": posy_key,
        "x": posx_key,
        "y": posy_key,
        "Slice n°": frame_key,
        "Track n°": track_key,
    }

    if "t" in traj:
        mapper.update({"t": frame_key})

    traj = traj.rename(mapper=mapper, axis=1)
    traj["TRACK_ID"] = traj["TRACK_ID"].fillna(-1)

    if traj["FRAME"].min() == 1:
        traj["FRAME"] = traj["FRAME"] - 1

    skel = sio.Skeleton(nodes=["centroid"])

    tracks = {}
    lfs = []
    for frame_idx in sorted(traj["FRAME"].unique()):
        insts = []
        lf = traj[traj["FRAME"] == frame_idx]
        for inst_idx in sorted(lf["TRACK_ID"].unique()):
            if inst_idx not in tracks:
                if inst_idx != -1:
                    tracks[int(inst_idx)] = sio.Track(f"Track {int(inst_idx) + 1}")
                else:
                    tracks[int(inst_idx)] = sio.Track("Unassigned Track")

            track = tracks[int(inst_idx)]

            instance = lf[lf["TRACK_ID"] == inst_idx]
            pt = np.array(
                (instance["POSITION_X"].iloc[0], instance["POSITION_Y"].iloc[0])
            )
            if np.isnan(pt).all():
                print("Nan found")
            try:
                insts.append(
                    sio.Instance.from_numpy(
                        pt.reshape(1, 2), skeleton=skel, track=track
                    )
                )
            except Exception as e:
                print(inst_idx)
                raise e
        lfs.append(sio.LabeledFrame(video=vid, frame_idx=frame_idx, instances=insts))

    labels = sio.Labels(lfs)
    labels.videos[0].backend.filename = str(Path(video_path).with_suffix(".mp4"))
    return labels


def tif2mp4(video_path: str, out_dir: str = ".", fps: int = 30) -> bool:
    """Convert a TIF video to MP4 format.

    Args:
        video_path: Path to the TIF file.
        out_dir: Output directory for the MP4 file.
        fps: Frames per second for the output video.

    Returns:
        True on success.
    """
    frames = imread(video_path)
    # Normalize to uint8 if needed (e.g., 16-bit microscopy images)
    if frames.dtype != np.uint8:
        fmin = frames.min()
        fmax = frames.max()
        if fmax > fmin:
            frames = ((frames - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            frames = np.zeros_like(frames, dtype=np.uint8)
    with imageio.get_writer(
        f"{out_dir}/{Path(video_path).stem}.mp4", fps=fps, macro_block_size=1
    ) as writer:
        for frame in tqdm(frames):
            writer.append_data(frame)
    return True


def nd2mp4(video_path: str, out_dir: str = ".", fps: int = 30) -> bool:
    """Convert an ND2 video to MP4 format.

    Args:
        video_path: Path to the ND2 file.
        out_dir: Output directory for the MP4 file.
        fps: Frames per second for the output video.

    Returns:
        True on success.
    """
    from nd2reader import ND2Reader

    with (
        ND2Reader(video_path) as nd2,
        imageio.get_writer(
            f"{out_dir}/{Path(video_path).stem}.mp4", fps=fps, macro_block_size=1
        ) as mp4,
    ):
        for frame in tqdm(nd2):
            mp4.append_data(frame)

    return True


def tif2npy(video_path: str, save: bool = True, out_dir: str = ".") -> np.ndarray:
    """Convert a TIF video to a NumPy array.

    Args:
        video_path: Path to the TIF file.
        save: Whether to save the array to disk.
        out_dir: Output directory for the .npy file.

    Returns:
        The video as a NumPy array.
    """
    vid = imread(video_path)
    vid = np.expand_dims(vid, -1)
    if save:
        print(f"Saving to {out_dir}/{Path(video_path).stem}.npy")
        np.save(f"{out_dir}/{Path(video_path).stem}.npy", vid)
    return vid


def convert_trackmate(
    label_files: list[str],
    vid_files: list[str],
    out_dir: str = ".",
    to_npy: bool = False,
    to_mp4: bool = False,
) -> None:
    """Convert TrackMate outputs to .slp files.

    Args:
        label_files: Paths to TrackMate CSV/XLSX label files.
        vid_files: Paths to video files (TIF, ND2, etc.).
        out_dir: Output directory for converted files.
        to_npy: Convert TIF videos to .npy format.
        to_mp4: Convert videos to .mp4 format.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if vid_files:
        for vid in vid_files:
            suffix = Path(vid).suffix
            if ".tif" in suffix:
                if to_npy:
                    print(f"Converting {Path(vid).stem} to .npy")
                    tif2npy(vid, out_dir=out_dir)
                elif to_mp4:
                    print(f"Converting {Path(vid).stem} to .mp4")
                    tif2mp4(vid, out_dir=out_dir)
            elif ".nd2" in suffix:
                if to_mp4:
                    print(f"Converting {Path(vid).stem} to .mp4")
                    nd2mp4(vid, out_dir=out_dir)
                elif to_npy:
                    print(
                        "`--to-npy` flag is currently not compatible with .nd2. "
                        "Use `--to-mp4` instead!"
                    )
                    continue
            else:
                print(f"Unknown file type, {suffix}, skipping!")
                continue

    if label_files and vid_files and len(label_files) > 0 and len(vid_files) > 0:
        for csv_file, tif in tqdm(zip(label_files, vid_files)):
            print(f"Converting {Path(csv_file).stem}\t{Path(tif).stem} to .slp")
            labels = make_labels(tif, csv_file)
            print(f"Saving to {out_dir}/{Path(tif).stem}.slp")
            sio.save_slp(labels, f"{out_dir}/{Path(tif).stem}.slp")
    else:
        raise ValueError("Either labels files or video files are missing!")
