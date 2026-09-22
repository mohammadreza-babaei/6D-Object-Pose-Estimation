"""Portable project paths and default experiment locations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "linemod" / "Linemod_preprocessed"
YOLO_DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_linemod"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUNS_ROOT = PROJECT_ROOT / "runs"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
