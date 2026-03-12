"""DREEM CLI - Global Tracking Transformer for biological multi-object tracking."""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

warnings.filterwarnings("ignore", message=".*num_workers.*")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(message)s",
)

import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from dreem.version import __version__  # noqa: E402

app = typer.Typer(
    name="dreem",
    help="DREEM: Global Tracking Transformer for biological multi-object tracking.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger("dreem.cli")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"dreem {__version__}")
        raise typer.Exit()


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def load_default_config(command: str):
    """Load default config for a command."""
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent / "configs" / "defaults" / f"{command}.yaml"
    return OmegaConf.load(config_path)


def build_config(
    command: str,
    config_file: Path | None,
    overrides: list[str] | None,
    **cli_args,
):
    """Build config with priority: defaults < config_file < cli_args."""
    from omegaconf import OmegaConf

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


def _resolve_accelerator(device: str, gpu: bool | None) -> str:
    """Resolve accelerator from --device and deprecated --gpu flags.

    Args:
        device: Value from --device flag (e.g. "auto", "gpu", "cpu", "mps").
        gpu: Value from deprecated --gpu/--no-gpu flag, or None if not provided.

    Returns:
        Resolved accelerator string.
    """
    if gpu is not None:
        if device != "auto":
            warnings.warn(
                "Both --device and --gpu were provided; --device takes precedence",
                UserWarning,
                stacklevel=3,
            )
            return device
        warnings.warn(
            "--gpu/--no-gpu is deprecated, use --device instead",
            FutureWarning,
            stacklevel=3,
        )
        return "cpu" if gpu is False else "gpu"
    return device


def _strip_training_config_sections(cfg):
    """Remove training-only config sections for inference/eval.

    Strips sections like optimizer, scheduler, loss, etc. that are present
    in pretrained model configs but irrelevant (and potentially harmful)
    during inference.
    """
    from omegaconf import open_dict

    from dreem.io.config import _TRAINING_ONLY_CONFIG_SECTIONS

    with open_dict(cfg):
        for section in _TRAINING_ONLY_CONFIG_SECTIONS:
            if section in cfg:
                del cfg[section]
    return cfg


def _flatten_config(cfg, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten nested OmegaConf config into dot-notation keys."""
    from omegaconf import OmegaConf

    items = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if OmegaConf.is_dict(value):
            items.update(_flatten_config(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def print_config(
    cfg,
    title: str = "Configuration",
    save_path: Path | None = None,
) -> None:
    """Print config summary as Rich table and optionally save to YAML."""
    from omegaconf import OmegaConf
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    # Save to YAML if save_path is provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            OmegaConf.save(cfg, f)
        console.print(f"[green]Configuration saved to: {save_path}[/green]")

    # Show only key settings (backward compatible)
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


def _create_inference_command(mode: str):
    """Factory function to create track/eval commands with shared signature."""

    def command(
        input_path: Annotated[Path, typer.Argument(help="Input data directory")],
        checkpoint: Annotated[
            str,
            typer.Option(
                "--checkpoint",
                "-ckpt",
                help="Model checkpoint: local path, shortname (animals, microscopy), HuggingFace repo (org/repo), or HuggingFace URL",
            ),
        ],
        output: Annotated[
            Path, typer.Option("--output", "-o", help="Output directory")
        ],
        crop_size: Annotated[
            int | None,
            typer.Option(
                "--crop-size",
                "-cs",
                help="Size of bounding box to crop each instance (pixels)",
            ),
        ],
        slp_files: Annotated[
            list[Path] | None,
            typer.Option("--slp-file", "-slp", help="Path to SLEAP label files"),
        ] = None,
        video_files: Annotated[
            list[Path] | None,
            typer.Option("--video-file", "-vid", help="Path to video files"),
        ] = None,
        video_type: Annotated[
            str | None,
            typer.Option(
                "--video-type",
                "-vt",
                help="Raw video file extension: For masks, use tif or tiff. For videos, use mp4, avi, mov, mkv, wmv, m4v, m4a",
            ),
        ] = "mp4",
        anchor: Annotated[
            str | None,
            typer.Option(
                "--anchor", "-a", help="Name of anchor keypoint e.g. 'centroid'"
            ),
        ] = "centroid",
        clip_length: Annotated[
            int | None,
            typer.Option("--clip-length", "-cl", help="Number of frames per batch"),
        ] = 32,
        max_detection_overlap: Annotated[
            float | None,
            typer.Option(
                "--max-detection-overlap",
                "-di",
                help="IOU threshold above which detections are considered duplicates",
            ),
        ] = None,
        dilation_radius: Annotated[
            int | None,
            typer.Option(
                "--dilation-radius",
                "-dr",
                help="Size of mask around the keypoint (pixels) to mask out background",
            ),
        ] = None,
        confidence_threshold: Annotated[
            float | None,
            typer.Option(
                "--confidence-threshold",
                "-conf",
                help="Threshold below which frames will be flagged as a potential error",
            ),
        ] = 0,
        iou_mode: Annotated[
            str | None, typer.Option("--iou-mode", "-iou", help="IOU mode (mult/add)")
        ] = "mult",
        max_dist: Annotated[
            float | None, typer.Option("--max-dist", "-md", help="Max center distance")
        ] = None,
        max_dist_multiplier: Annotated[
            float | None,
            typer.Option(
                "--max-dist-multiplier",
                "-mdm",
                help="Multiplier applied to the max-dist penalty (default: 1.0 if max_dist is provided)",
            ),
        ] = None,
        max_gap: Annotated[
            int | None, typer.Option("--max-gap", "-mg", help="Max frame gap")
        ] = None,
        overlap_thresh: Annotated[
            float | None,
            typer.Option("--overlap-thresh", "-ot", help="Overlap threshold"),
        ] = None,
        mult_thresh: Annotated[
            bool | None,
            typer.Option("--mult-thresh", "-mt", help="Use multiplicative threshold"),
        ] = None,
        max_angle: Annotated[
            float | None,
            typer.Option("--max-angle", "-ma", help="Max angle difference"),
        ] = None,
        max_tracks: Annotated[
            int | None, typer.Option("--max-tracks", "-mx", help="Max number of tracks")
        ] = None,
        front_nodes: Annotated[
            list[str] | None,
            typer.Option("--front-node", "-fn", help="Front nodes for orientation"),
        ] = None,
        back_nodes: Annotated[
            list[str] | None,
            typer.Option("--back-node", "-bn", help="Back nodes for orientation"),
        ] = None,
        save_meta: Annotated[
            bool, typer.Option("--save-meta", "-sm", help="Save frame metadata")
        ] = False,
        device: Annotated[
            str,
            typer.Option("--device", help="Accelerator: auto, gpu, cpu, mps"),
        ] = "auto",
        gpu: Annotated[
            bool | None,
            typer.Option(
                "--gpu/--no-gpu",
                "-g",
                help="[deprecated] Use --device instead",
                hidden=True,
            ),
        ] = None,
        limit_batches: Annotated[
            float | None,
            typer.Option(
                "--limit-batches",
                help="Limit inference to N batches (for debugging)",
            ),
        ] = None,
        config: Annotated[
            Path | None,
            typer.Option("--config", "-c", help="Config file (overrides defaults)"),
        ] = None,
        set_: Annotated[
            list[str] | None,
            typer.Option(
                "--set", "-s", help="Config overrides (e.g., tracker.decay_time=0.9)"
            ),
        ] = None,
        quiet: Annotated[
            bool, typer.Option("--quiet", "-q", help="Suppress progress output")
        ] = False,
        verbose: Annotated[
            bool, typer.Option("--verbose", help="Enable verbose logging")
        ] = False,
    ) -> None:
        """Shared implementation for track and eval commands."""
        from dreem.io.pretrained import (
            _is_hf_repo_id,
            _parse_hf_url,
            is_pretrained_shortname,
            resolve_checkpoint,
            resolve_config,
        )

        if verbose:
            logging.getLogger("dreem").setLevel(logging.INFO)

        is_hf_model = (
            is_pretrained_shortname(checkpoint)
            or _is_hf_repo_id(checkpoint)
            or _parse_hf_url(checkpoint) is not None
        )
        if is_hf_model:
            console.print(f"[cyan]Resolving pretrained model: {checkpoint}[/cyan]")
            # Auto-load config from HuggingFace if user didn't provide one
            if config is None:
                hf_config = resolve_config(checkpoint)
                if hf_config is not None:
                    config = Path(hf_config)
                    console.print(f"[green]Using config: {config}[/green]")
            checkpoint = resolve_checkpoint(checkpoint)
            console.print(f"[green]Using checkpoint: {checkpoint}[/green]")
        elif not Path(checkpoint).exists():
            console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
            raise typer.Exit(1)

        if not input_path.exists():
            console.print(f"[red]Error: Input path not found: {input_path}[/red]")
            raise typer.Exit(1)

        accelerator = _resolve_accelerator(device, gpu)

        cli_overrides = {
            "ckpt_path": str(checkpoint),
            "outdir": str(output),
            "dataset.test_dataset.dir.path": str(input_path),
            "dataset.test_dataset.anchors": anchor,
            "dataset.test_dataset.clip_length": clip_length,
            "dataset.test_dataset.slp_files": [str(f) for f in slp_files]
            if slp_files
            else None,
            "dataset.test_dataset.video_files": [str(f) for f in video_files]
            if video_files
            else None,
            "dataset.test_dataset.dir.labels_suffix": f".{video_type}"
            if "tif" in video_type
            else ".slp",
            "dataset.test_dataset.dir.vid_suffix": f".{video_type}",
            "tracker.iou": iou_mode,
            "tracker.max_center_dist": max_dist,
            "tracker.distance_penalty_multiplier": max_dist_multiplier,
            "tracker.overlap_thresh": overlap_thresh,
            "dataset.test_dataset.crop_size": crop_size,
            "dataset.test_dataset.max_detection_overlap": max_detection_overlap,
            "dataset.test_dataset.dilation_radius_px": dilation_radius,
            "tracker.confidence_threshold": confidence_threshold,
            "tracker.max_gap": max_gap,
            "tracker.mult_thresh": mult_thresh,
            "tracker.max_angle_diff": max_angle,
            "tracker.max_tracks": max_tracks,
            "tracker.front_nodes": list(front_nodes) if front_nodes else None,
            "tracker.back_nodes": list(back_nodes) if back_nodes else None,
            "save_frame_meta": save_meta,
            "trainer.accelerator": accelerator,
        }

        if limit_batches is not None:
            cli_overrides["trainer.limit_predict_batches"] = limit_batches

        cfg = build_config("track", config, set_, **cli_overrides)
        _strip_training_config_sections(cfg)

        config_title = (
            "Track Configuration" if mode == "track" else "Eval Configuration"
        )

        # Determine save path for config YAML
        save_path = None
        outdir = Path(cfg.outdir) if "outdir" in cfg else Path("./results")
        outdir.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
        save_path = outdir / f"config.{mode}.{timestamp}.yaml"

        if not quiet:
            print_config(cfg, config_title, save_path=save_path)

        if mode == "track":
            from dreem.inference.track import run as run_tracking

            run_tracking(cfg)

        else:
            from dreem.inference.eval import run as run_eval

            run_eval(cfg)
            console.print("[green]Evaluation complete.[/green]")

    # Set function metadata for help text
    command.__name__ = mode
    if mode == "track":
        command.__doc__ = "Run tracking on a dataset."
    else:
        command.__doc__ = "Evaluate a trained DREEM model against ground truth."

    return command


# Register commands using factory function - signature defined once above
track = app.command()(_create_inference_command("track"))
eval_cmd = app.command(name="eval")(_create_inference_command("eval"))


@app.command()
def convert(
    format: Annotated[
        str, typer.Argument(help="Source format to convert from (e.g. 'trackmate')")
    ],
    labels: Annotated[
        list[Path] | None,
        typer.Option("--labels", "-l", help="Paths to label files (CSV/XLSX)"),
    ] = None,
    videos: Annotated[
        list[Path] | None,
        typer.Option("--videos", "-v", help="Paths to video files (TIF, ND2, etc.)"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for converted files"),
    ] = Path("."),
    to_npy: Annotated[
        bool,
        typer.Option("--to-npy", "-n", help="Convert TIF videos to .npy format"),
    ] = False,
    to_mp4: Annotated[
        bool,
        typer.Option("--to-mp4", "-m", help="Convert videos to .mp4 format"),
    ] = False,
) -> None:
    """Convert external tracking formats to .slp files.

    Usage: dreem convert trackmate -l labels1.csv -l labels2.csv -v video1.tif -v video2.tif
    """
    supported_formats = ["trackmate"]
    if format not in supported_formats:
        console.print(
            f"[red]Error: Unknown format '{format}'. "
            f"Supported formats: {', '.join(supported_formats)}[/red]"
        )
        raise typer.Exit(1)

    if not labels or not videos:
        console.print("[red]Error: Both --labels and --videos are required.[/red]")
        raise typer.Exit(1)

    if format == "trackmate":
        from dreem.utils.convert import convert_trackmate

        convert_trackmate(
            label_files=[str(f) for f in labels],
            vid_files=[str(f) for f in videos],
            out_dir=str(output),
            to_npy=to_npy,
            to_mp4=to_mp4,
        )

    console.print("[green]Conversion complete.[/green]")


@app.command()
def train(
    train_dir: Annotated[Path, typer.Argument(help="Training data directory")],
    val_dir: Annotated[
        Path, typer.Option("--val-dir", "-vd", help="Validation data directory")
    ],
    crop_size: Annotated[
        int | None, typer.Option("--crop-size", "-cs", help="Crop size")
    ],
    video_type: Annotated[
        str | None,
        typer.Option(
            "--video-type",
            "-vt",
            help="Raw video file extension: For masks, use tif or tiff. For videos, use mp4, avi, mov, mkv, wmv, m4v, m4a",
        ),
    ] = "mp4",
    epochs: Annotated[
        int | None, typer.Option("--epochs", "-e", help="Max epochs")
    ] = 20,
    lr: Annotated[float | None, typer.Option("--lr", help="Learning rate")] = 0.0001,
    d_model: Annotated[
        int | None, typer.Option("--d-model", help="Model dimension")
    ] = 128,
    nhead: Annotated[
        int | None, typer.Option("--nhead", help="Number of attention heads")
    ] = 1,
    num_encoder_layers: Annotated[
        int | None, typer.Option("--encoder-layers", help="Encoder layers")
    ] = 1,
    num_decoder_layers: Annotated[
        int | None, typer.Option("--decoder-layers", help="Decoder layers")
    ] = 1,
    anchor: Annotated[
        str | None, typer.Option("--anchor", "-a", help="Anchor type")
    ] = "centroid",
    clip_length: Annotated[
        int | None, typer.Option("--clip-length", "-cl", help="Clip length")
    ] = 32,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Accelerator: auto, gpu, cpu, mps"),
    ] = "auto",
    gpu: Annotated[
        bool | None,
        typer.Option(
            "--gpu/--no-gpu",
            "-g",
            help="[deprecated] Use --device instead",
            hidden=True,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Config file (overrides defaults)"),
    ] = None,
    logger: Annotated[
        str | None,
        typer.Option(
            "--logger",
            "-l",
            help="Logger type (any Lightning logger e.g. WandbLogger, TensorBoardLogger)",
        ),
    ] = None,
    run_name: Annotated[
        str | None,
        typer.Option(
            "--run-name",
            "-rn",
            help="Name of model run (creates subdirectory ./models/run_name)",
        ),
    ] = "dreem_train",
    set_: Annotated[
        list[str] | None, typer.Option("--set", "-s", help="Config overrides")
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress progress output")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable verbose logging")
    ] = False,
) -> None:
    """Train a DREEM model."""
    from omegaconf import OmegaConf

    if verbose:
        logging.getLogger("dreem").setLevel(logging.INFO)

    if not train_dir.exists():
        console.print(f"[red]Error: Training directory not found: {train_dir}[/red]")
        raise typer.Exit(1)

    if not val_dir.exists():
        console.print(f"[red]Error: Validation directory not found: {val_dir}[/red]")
        raise typer.Exit(1)

    accelerator = _resolve_accelerator(device, gpu)

    cli_overrides = {
        "dataset.train_dataset.dir.path": str(train_dir),
        "dataset.val_dataset.dir.path": str(val_dir),
        "dataset.train_dataset.dir.labels_suffix": f".{video_type}"
        if "tif" in video_type
        else ".slp",
        "dataset.val_dataset.dir.labels_suffix": f".{video_type}"
        if "tif" in video_type
        else ".slp",
        "dataset.train_dataset.dir.vid_suffix": f".{video_type}",
        "dataset.val_dataset.dir.vid_suffix": f".{video_type}",
        "trainer.max_epochs": epochs,
        "optimizer.lr": lr,
        "model.d_model": d_model,
        "model.nhead": nhead,
        "model.num_encoder_layers": num_encoder_layers,
        "model.num_decoder_layers": num_decoder_layers,
        "dataset.train_dataset.anchors": anchor,
        "dataset.val_dataset.anchors": anchor,
        "dataset.train_dataset.clip_length": clip_length,
        "dataset.train_dataset.crop_size": crop_size,
        "dataset.val_dataset.crop_size": crop_size,
        "logging.name": run_name,
        "logging.logger_type": logger,
        "trainer.accelerator": accelerator,
    }

    cfg = build_config("train", config, set_, **cli_overrides)

    # Determine save path for config YAML
    save_path = None
    save_dir = Path("./logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    run_name = OmegaConf.select(cfg, "logging.name") or "train"
    save_path = save_dir / f"config.{run_name}.{timestamp}.yaml"

    from dreem.training.train import run as run_training

    if not quiet:
        print_config(cfg, "Train Configuration", save_path=save_path)
        console.print("[cyan]Starting training...[/cyan]")

    run_training(cfg)
    console.print("[green]Training complete.[/green]")


@app.command()
def system() -> None:
    """Display system information and GPU status.

    Shows Python version, platform, PyTorch version, CUDA availability,
    driver version with compatibility check, GPU details, and package versions.
    """
    from dreem.system_info import print_system_info

    print_system_info()
