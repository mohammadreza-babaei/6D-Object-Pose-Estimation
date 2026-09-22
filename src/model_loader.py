"""Utilities for loading LineMOD 3D object models."""

from pathlib import Path

import torch


def load_ply_vertices(ply_path: str) -> torch.Tensor:
    """Load XYZ vertices from an ASCII PLY file."""

    path = Path(ply_path)

    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, "r") as file:
        lines = file.readlines()

    vertex_count = None
    header_end = None

    for index, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])

        if line.strip() == "end_header":
            header_end = index + 1
            break

    if vertex_count is None or header_end is None:
        raise ValueError("Invalid PLY file.")

    vertices = []

    for line in lines[header_end:header_end + vertex_count]:
        values = line.split()

        vertices.append([
            float(values[0]),
            float(values[1]),
            float(values[2]),
        ])

    return torch.tensor(vertices, dtype=torch.float32)