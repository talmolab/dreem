"""DREEM CLI - Global Tracking Transformer for biological multi-object tracking."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dreem.version import __version__

app = typer.Typer(
    name="dreem",
    help="DREEM: Global Tracking Transformer for biological multi-object tracking.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger("dreem.cli")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"dreem {__version__}")
        raise typer.Exit()


def get_timestamp() -> str:
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def load_default_config(command: str) -> DictConfig:
    """Load default config for a command."""
    config_path = Path(__file__).parent / "configs" / "defaults" / f"{command}.yaml"
    return OmegaConf.load(config_path)


def build_config(
    command: str,
    config_file: Path | None,
    overrides: list[str] | None,
    **cli_args,
) -> DictConfig:
    """Build config with priority: defaults < config_file < cli_args."""
    cfg = load_default_config(command)

    if config_file:
        user_cfg = OmegaConf.load(config_file)
        cfg = OmegaConf.merge(cfg, user_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    for key, value in cli_args.items():
        if value is not None:
            OmegaConf.update(cfg, key, value, merge=True)

    return cfg


def print_config(cfg: DictConfig, title: str = "Configuration") -> None:
    """Print config summary as Rich table."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    if OmegaConf.select(cfg, "ckpt_path"):
        table.add_row("Checkpoint:", str(cfg.ckpt_path))
    if OmegaConf.select(cfg, "outdir"):
        table.add_row("Output:", str(cfg.outdir))
    if OmegaConf.select(cfg, "dataset.test_dataset.dir.path"):
        table.add_row("Input:", str(cfg.dataset.test_dataset.dir.path))

    console.print(Panel(table, title=title))


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """DREEM: Global Tracking Transformer for biological multi-object tracking."""
    pass


@app.command()
def track(
    input_path: Annotated[Path, typer.Argument(help="Input data directory")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-ckpt", help="Model checkpoint path")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")],

    slp_files: Annotated[list[Path] | None, typer.Option("--slp-file", "-slp", help="SLEAP label files")] = None,
    video_files: Annotated[list[Path] | None, typer.Option("--video-file", "-vid", help="Video files")] = None,

    anchors: Annotated[str | None, typer.Option("--anchors", "-a", help="Anchor type")] = None,
    clip_length: Annotated[int | None, typer.Option("--clip-length", "-cl", help="Clip length")] = None,
    crop_size: Annotated[int | None, typer.Option("--crop-size", "-cs", help="Crop size")] = None,
    detection_iou: Annotated[float | None, typer.Option("--detection-iou", "-di", help="Detection IOU threshold")] = None,
    dilation_radius: Annotated[int | None, typer.Option("--dilation-radius", "-dr", help="Dilation radius in pixels")] = None,

    confidence: Annotated[float | None, typer.Option("--confidence", "-conf", help="Confidence threshold")] = None,
    iou_mode: Annotated[str | None, typer.Option("--iou-mode", "-iou", help="IOU mode (mult/add)")] = None,
    max_dist: Annotated[float | None, typer.Option("--max-dist", "-md", help="Max center distance")] = None,
    max_gap: Annotated[int | None, typer.Option("--max-gap", "-mg", help="Max frame gap")] = None,
    overlap_thresh: Annotated[float | None, typer.Option("--overlap-thresh", "-ot", help="Overlap threshold")] = None,
    mult_thresh: Annotated[bool | None, typer.Option("--mult-thresh", "-mt", help="Use multiplicative threshold")] = None,
    max_angle: Annotated[float | None, typer.Option("--max-angle", "-ma", help="Max angle difference")] = None,
    max_tracks: Annotated[int | None, typer.Option("--max-tracks", "-mx", help="Max number of tracks")] = None,
    front_nodes: Annotated[list[str] | None, typer.Option("--front-node", "-fn", help="Front nodes for orientation")] = None,
    back_nodes: Annotated[list[str] | None, typer.Option("--back-node", "-bn", help="Back nodes for orientation")] = None,

    save_meta: Annotated[bool, typer.Option("--save-meta", "-sm", help="Save frame metadata")] = False,

    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file (overrides defaults)")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", "-s", help="Config overrides (e.g., tracker.decay_time=0.9)")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
) -> None:
    """Run tracking inference on a video dataset."""
    if not checkpoint.exists():
        console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    if not input_path.exists():
        console.print(f"[red]Error: Input path not found: {input_path}[/red]")
        raise typer.Exit(1)

    cli_overrides = {
        "ckpt_path": str(checkpoint),
        "outdir": str(output),
        "dataset.test_dataset.dir.path": str(input_path),
        "dataset.test_dataset.anchors": anchors,
        "dataset.test_dataset.clip_length": clip_length,
        "dataset.test_dataset.crop_size": crop_size,
        "dataset.test_dataset.detection_iou_threshold": detection_iou,
        "dataset.test_dataset.dilation_radius_px": dilation_radius,
        "dataset.test_dataset.slp_files": [str(f) for f in slp_files] if slp_files else None,
        "dataset.test_dataset.video_files": [str(f) for f in video_files] if video_files else None,
        "tracker.confidence_threshold": confidence,
        "tracker.iou": iou_mode,
        "tracker.max_center_dist": max_dist,
        "tracker.max_gap": max_gap,
        "tracker.overlap_thresh": overlap_thresh,
        "tracker.mult_thresh": mult_thresh,
        "tracker.max_angle_diff": max_angle,
        "tracker.max_tracks": max_tracks,
        "tracker.front_nodes": list(front_nodes) if front_nodes else None,
        "tracker.back_nodes": list(back_nodes) if back_nodes else None,
        "save_frame_meta": save_meta,
    }

    cfg = build_config("track", config, set_, **cli_overrides)

    if not quiet:
        print_config(cfg, "Track Configuration")

    _run_tracking(cfg, quiet)


def _run_tracking(cfg: DictConfig, quiet: bool = False) -> None:
    """Execute tracking - mirrors dreem/inference/track.py:run()."""
    import h5py
    import numpy as np
    import pytorch_lightning as pl
    import sleap_io as sio
    from sleap_io.model.suggestions import SuggestionFrame
    import tifffile
    from tqdm import tqdm

    from dreem.datasets import CellTrackingDataset
    from dreem.io import Config, Frame
    from dreem.io.flags import FrameFlagCode
    from dreem.models import GTRRunner

    pred_cfg = Config(cfg)

    model = GTRRunner.load_from_checkpoint(cfg.ckpt_path, strict=False)
    overrides_dict = model.setup_tracking(pred_cfg, mode="inference")

    labels_files, vid_files = pred_cfg.get_data_paths(
        "test", pred_cfg.cfg.dataset.test_dataset
    )
    trainer = pred_cfg.get_trainer()
    outdir = cfg.outdir if "outdir" in cfg else "./results"
    os.makedirs(outdir, exist_ok=True)

    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = pred_cfg.get_dataset(
            label_files=[label_file],
            vid_files=[vid_file],
            mode="test",
            overrides=overrides_dict,
        )
        dataloader = pred_cfg.get_dataloader(dataset, mode="test")

        if isinstance(vid_file, list):
            save_file_name = vid_file[0].split("/")[-2]
        else:
            save_file_name = vid_file

        if isinstance(dataset, CellTrackingDataset):
            preds = trainer.predict(model, dataloader)
            pred_imgs = []
            for batch in preds:
                for frame in batch:
                    frame_masks = []
                    for instance in frame.instances:
                        mask = instance.mask.cpu().numpy()
                        track_id = instance.pred_track_id.cpu().numpy().item()
                        mask = mask.astype(np.uint8)
                        mask[mask != 0] = track_id
                        frame_masks.append(mask)
                    
                    if frame_masks:
                        frame_mask = np.max(frame_masks, axis=0)
                    else:
                        # Handle empty instances case - create zero mask with image dimensions
                        img_shape = frame.img_shape
                        if len(img_shape) == 3:
                            # img_shape is (C, H, W), extract spatial dimensions
                            _, height, width = img_shape
                        elif len(img_shape) == 2:
                            # img_shape is (H, W)
                            height, width = img_shape
                        frame_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    pred_imgs.append(frame_mask)
            pred_imgs = np.stack(pred_imgs)
            outpath = os.path.join(
                outdir,
                f"{Path(save_file_name).stem}.dreem_inference.{get_timestamp()}.tif",
            )
            tifffile.imwrite(outpath, pred_imgs.astype(np.uint16))
        else:
            save_frame_meta = overrides_dict.get("save_frame_meta", False)
            if save_frame_meta:
                h5_path = os.path.join(
                    outdir,
                    f"{dataloader.dataset.slp_files[0].split('/')[-1].replace('.slp', '')}_frame_meta.h5",
                )
                if os.path.exists(h5_path):
                    os.remove(h5_path)
                with h5py.File(h5_path, "a") as h5f:
                    pass

            suggestions = []
            preds = trainer.predict(model, dataloader)
            pred_slp = []
            tracks = {}

            iterator = tqdm(preds, desc="Saving .slp and frame metadata") if not quiet else preds
            for batch in iterator:
                for frame in batch:
                    if frame.frame_id.item() == 0:
                        video = (
                            sio.Video(frame.video)
                            if isinstance(frame.video, str)
                            else sio.Video
                        )
                    if frame.has_flag(FrameFlagCode.LOW_CONFIDENCE):
                        suggestion = SuggestionFrame(
                            video=video, frame_idx=frame.frame_id.item()
                        )
                        suggestions.append(suggestion)
                    lf, tracks = frame.to_slp(tracks, video=video)
                    pred_slp.append(lf)
                    if save_frame_meta:
                        _store_frame_metadata(frame, h5_path)

            pred_slp = sio.Labels(pred_slp, suggestions=suggestions)
            outpath = os.path.join(
                outdir,
                f"{Path(save_file_name).stem}.dreem_inference.{get_timestamp()}.slp",
            )
            pred_slp.save(outpath)

    console.print(f"[green]Results saved to {outdir}[/green]")


def _store_frame_metadata(frame, h5_path: str) -> None:
    """Store frame metadata to HDF5."""
    import h5py
    with h5py.File(h5_path, "a") as h5f:
        frame_meta_group = h5f.require_group("frame_meta")
        frame = frame.to("cpu")
        _ = frame.to_h5(
            frame_meta_group,
            frame.get_gt_track_ids().cpu().numpy(),
            save={"features": True, "crop": True},
        )


@app.command()
def eval(
    input_path: Annotated[Path, typer.Argument(help="Input data directory")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-ckpt", help="Model checkpoint path")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")],

    slp_files: Annotated[list[Path] | None, typer.Option("--slp-file", "-slp", help="SLEAP label files")] = None,
    video_files: Annotated[list[Path] | None, typer.Option("--video-file", "-vid", help="Video files")] = None,

    anchors: Annotated[str | None, typer.Option("--anchors", "-a", help="Anchor type")] = None,
    clip_length: Annotated[int | None, typer.Option("--clip-length", "-cl", help="Clip length")] = None,

    iou_mode: Annotated[str | None, typer.Option("--iou-mode", "-iou", help="IOU mode (mult/add)")] = None,
    max_dist: Annotated[float | None, typer.Option("--max-dist", "-md", help="Max center distance")] = None,
    overlap_thresh: Annotated[float | None, typer.Option("--overlap-thresh", "-ot", help="Overlap threshold")] = None,

    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file (overrides defaults)")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", "-s", help="Config overrides")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
) -> None:
    """Evaluate a trained DREEM model against ground truth."""
    if not checkpoint.exists():
        console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    if not input_path.exists():
        console.print(f"[red]Error: Input path not found: {input_path}[/red]")
        raise typer.Exit(1)

    cli_overrides = {
        "ckpt_path": str(checkpoint),
        "outdir": str(output),
        "dataset.test_dataset.dir.path": str(input_path),
        "dataset.test_dataset.anchors": anchors,
        "dataset.test_dataset.clip_length": clip_length,
        "dataset.test_dataset.slp_files": [str(f) for f in slp_files] if slp_files else None,
        "dataset.test_dataset.video_files": [str(f) for f in video_files] if video_files else None,
        "tracker.iou": iou_mode,
        "tracker.max_center_dist": max_dist,
        "tracker.overlap_thresh": overlap_thresh,
    }

    cfg = build_config("track", config, set_, **cli_overrides)

    if not quiet:
        print_config(cfg, "Eval Configuration")

    _run_eval(cfg, quiet)


def _run_eval(cfg: DictConfig, quiet: bool = False) -> None:
    """Execute evaluation - mirrors dreem/inference/eval.py:run()."""
    from dreem.io import Config
    from dreem.models import GTRRunner

    eval_cfg = Config(cfg)

    model = GTRRunner.load_from_checkpoint(cfg.ckpt_path, strict=False)
    overrides_dict = model.setup_tracking(eval_cfg, mode="eval")

    if not quiet:
        console.print(f"[cyan]Saving results to {model.test_results['save_path']}[/cyan]")

    labels_files, vid_files = eval_cfg.get_data_paths(
        "test", eval_cfg.cfg.dataset.test_dataset
    )
    trainer = eval_cfg.get_trainer()

    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = eval_cfg.get_dataset(
            label_files=[label_file],
            vid_files=[vid_file],
            mode="test",
            overrides=overrides_dict,
        )
        dataloader = eval_cfg.get_dataloader(dataset, mode="test")
        _ = trainer.test(model, dataloader)

    console.print("[green]Evaluation complete.[/green]")


@app.command()
def train(
    train_dir: Annotated[Path, typer.Argument(help="Training data directory")],
    val_dir: Annotated[Path, typer.Option("--val-dir", "-vd", help="Validation data directory")],

    epochs: Annotated[int | None, typer.Option("--epochs", "-e", help="Max epochs")] = None,
    lr: Annotated[float | None, typer.Option("--lr", help="Learning rate")] = None,

    d_model: Annotated[int | None, typer.Option("--d-model", help="Model dimension")] = None,
    nhead: Annotated[int | None, typer.Option("--nhead", help="Number of attention heads")] = None,
    num_encoder_layers: Annotated[int | None, typer.Option("--encoder-layers", help="Encoder layers")] = None,
    num_decoder_layers: Annotated[int | None, typer.Option("--decoder-layers", help="Decoder layers")] = None,

    anchors: Annotated[str | None, typer.Option("--anchors", "-a", help="Anchor type")] = None,
    clip_length: Annotated[int | None, typer.Option("--clip-length", "-cl", help="Clip length")] = None,
    crop_size: Annotated[int | None, typer.Option("--crop-size", "-cs", help="Crop size")] = None,

    log_dir: Annotated[Path | None, typer.Option("--log-dir", "-ld", help="Log directory")] = None,
    run_name: Annotated[str | None, typer.Option("--run-name", "-rn", help="Run name for logging")] = None,

    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file (overrides defaults)")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", "-s", help="Config overrides")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
) -> None:
    """Train a DREEM model."""
    if not train_dir.exists():
        console.print(f"[red]Error: Training directory not found: {train_dir}[/red]")
        raise typer.Exit(1)

    if not val_dir.exists():
        console.print(f"[red]Error: Validation directory not found: {val_dir}[/red]")
        raise typer.Exit(1)

    cli_overrides = {
        "dataset.train_dataset.dir.path": str(train_dir),
        "dataset.val_dataset.dir.path": str(val_dir),
        "trainer.max_epochs": epochs,
        "optimizer.lr": lr,
        "model.d_model": d_model,
        "model.nhead": nhead,
        "model.num_encoder_layers": num_encoder_layers,
        "model.num_decoder_layers": num_decoder_layers,
        "dataset.train_dataset.anchors": anchors,
        "dataset.val_dataset.anchors": anchors,
        "dataset.train_dataset.clip_length": clip_length,
        "dataset.train_dataset.crop_size": crop_size,
        "dataset.val_dataset.crop_size": crop_size,
        "logging.save_dir": str(log_dir) if log_dir else None,
        "logging.name": run_name,
    }

    cfg = build_config("train", config, set_, **cli_overrides)

    if not quiet:
        print_config(cfg, "Train Configuration")

    _run_training(cfg, quiet)


def _run_training(cfg: DictConfig, quiet: bool = False) -> None:
    """Execute training - mirrors dreem/training/train.py:run()."""
    import torch
    import pytorch_lightning as pl

    from dreem.datasets import TrackingDataset
    from dreem.io import Config

    torch.set_float32_matmul_precision("medium")
    train_cfg = Config(cfg)

    model = train_cfg.get_model()

    train_dataset = train_cfg.get_dataset(mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    dataset = TrackingDataset(train_dl=train_dataloader, val_dl=val_dataloader)

    model = train_cfg.get_gtr_runner()
    run_logger = train_cfg.get_logger()

    callbacks = []
    callbacks.extend(train_cfg.get_checkpointing())
    callbacks.append(pl.callbacks.LearningRateMonitor())

    early_stopping = train_cfg.get_early_stopping()
    if early_stopping is not None:
        callbacks.append(early_stopping)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer = train_cfg.get_trainer(
        callbacks,
        run_logger,
        accelerator=accelerator,
        devices=devices,
    )

    if not quiet:
        console.print("[cyan]Starting training...[/cyan]")

    trainer.fit(model, dataset)

    console.print("[green]Training complete.[/green]")


@app.command()
def system() -> None:
    """Show system information and GPU status."""
    import torch
    import sys

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Python:", sys.version.split()[0])
    table.add_row("PyTorch:", torch.__version__)
    table.add_row("CUDA available:", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("CUDA version:", torch.version.cuda or "N/A")
        table.add_row("GPU:", torch.cuda.get_device_name(0))
        table.add_row("GPU count:", str(torch.cuda.device_count()))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        table.add_row("MPS available:", "True")

    console.print(Panel(table, title="System Information"))
