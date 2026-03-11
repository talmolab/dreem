"""Data loading and preprocessing."""

__all__ = [
    "BaseDataset",
    "CellTrackingDataset",
    "MicroscopyDataset",
    "SleapDataset",
    "TrackingDataset",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseDataset": ("dreem.datasets.base_dataset", "BaseDataset"),
    "CellTrackingDataset": (
        "dreem.datasets.cell_tracking_dataset",
        "CellTrackingDataset",
    ),
    "MicroscopyDataset": ("dreem.datasets.microscopy_dataset", "MicroscopyDataset"),
    "SleapDataset": ("dreem.datasets.sleap_dataset", "SleapDataset"),
    "TrackingDataset": ("dreem.datasets.tracking_dataset", "TrackingDataset"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dreem.datasets' has no attribute {name!r}")
