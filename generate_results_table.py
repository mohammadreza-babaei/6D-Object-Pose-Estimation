"""Generate a report-ready summary from saved experiment results."""

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
POSE_RESULTS_PATH = PROJECT_ROOT / "outputs" / "rgb_vs_rgbd.json"
YOLO_RESULTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "linemod_yolo"
    / "results.csv"
)
OUTPUT_JSON_PATH = PROJECT_ROOT / "outputs" / "final_results.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "final_results.csv"


def load_pose_results() -> tuple[dict[str, Any] | None, list[str]]:
    """Load RGB and RGB-D pose results when the saved JSON is available."""

    missing_metrics: list[str] = []
    if not POSE_RESULTS_PATH.exists():
        missing_metrics.extend(["RGB ADD", "RGB-D ADD"])
        return None, missing_metrics

    with POSE_RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    rgb_add = results.get("rgb", {}).get("add")
    rgbd_add = results.get("rgbd", {}).get("add")
    if rgb_add is None:
        missing_metrics.append("RGB ADD")
    if rgbd_add is None:
        missing_metrics.append("RGB-D ADD")

    return results, missing_metrics


def load_best_yolo_row() -> tuple[dict[str, Any] | None, list[str]]:
    """Load the recorded YOLO row with the highest mAP50-95."""

    required_metrics = ["mAP50", "mAP50-95"]
    if not YOLO_RESULTS_PATH.exists():
        return None, required_metrics

    with YOLO_RESULTS_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None, required_metrics

    map_column = "metrics/mAP50-95(B)"
    map50_column = "metrics/mAP50(B)"
    valid_rows = [
        row
        for row in rows
        if row.get(map_column, "").strip() and row.get(map50_column, "").strip()
    ]
    if not valid_rows:
        return None, required_metrics

    best_row = max(valid_rows, key=lambda row: float(row[map_column]))
    return best_row, []


def make_summary() -> dict[str, Any]:
    """Build a JSON-serializable final-results summary."""

    pose_results, missing_pose_metrics = load_pose_results()
    yolo_row, missing_yolo_metrics = load_best_yolo_row()
    missing_metrics = missing_yolo_metrics + missing_pose_metrics

    rgb_add = None
    rgbd_add = None
    add_improvement_percent = None
    if pose_results is not None:
        rgb_add = pose_results.get("rgb", {}).get("add")
        rgbd_add = pose_results.get("rgbd", {}).get("add")
        if rgb_add is not None and rgbd_add is not None and float(rgb_add) != 0:
            add_improvement_percent = (
                (float(rgb_add) - float(rgbd_add)) / float(rgb_add) * 100
            )

    yolo_summary = {
        "mAP50": float(yolo_row["metrics/mAP50(B)"]) if yolo_row else None,
        "mAP50-95": float(yolo_row["metrics/mAP50-95(B)"]) if yolo_row else None,
        "epoch": int(float(yolo_row["epoch"])) if yolo_row else None,
        "selection": "row with highest recorded mAP50-95",
        "source": str(YOLO_RESULTS_PATH.relative_to(PROJECT_ROOT)),
    }

    return {
        "object_id": pose_results.get("object_id") if pose_results else None,
        "split": pose_results.get("split") if pose_results else None,
        "detection": yolo_summary,
        "pose_estimation": {
            "unit": "millimeters",
            "RGB_ADD": rgb_add,
            "RGB-D_ADD": rgbd_add,
            "ADD_improvement_percent": add_improvement_percent,
            "source": str(POSE_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        },
        "missing_metrics": missing_metrics,
    }


def write_csv(summary: dict[str, Any]) -> None:
    """Write a flat report table suitable for spreadsheets."""

    detection = summary["detection"]
    pose = summary["pose_estimation"]
    rows = [
        {
            "category": "object detection",
            "metric": "mAP50",
            "value": detection["mAP50"],
            "unit": "score",
            "method": "YOLO",
            "status": "available" if detection["mAP50"] is not None else "missing",
        },
        {
            "category": "object detection",
            "metric": "mAP50-95",
            "value": detection["mAP50-95"],
            "unit": "score",
            "method": "YOLO",
            "status": "available" if detection["mAP50-95"] is not None else "missing",
        },
        {
            "category": "pose estimation",
            "metric": "ADD",
            "value": pose["RGB_ADD"],
            "unit": "mm",
            "method": "RGB",
            "status": "available" if pose["RGB_ADD"] is not None else "missing",
        },
        {
            "category": "pose estimation",
            "metric": "ADD",
            "value": pose["RGB-D_ADD"],
            "unit": "mm",
            "method": "RGB-D",
            "status": "available" if pose["RGB-D_ADD"] is not None else "missing",
        },
        {
            "category": "pose estimation",
            "metric": "ADD improvement",
            "value": pose["ADD_improvement_percent"],
            "unit": "%",
            "method": "RGB to RGB-D",
            "status": (
                "available"
                if pose["ADD_improvement_percent"] is not None
                else "missing"
            ),
        },
    ]

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summary = make_summary()
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(summary)
    print(f"Saved JSON summary: {OUTPUT_JSON_PATH}")
    print(f"Saved CSV summary: {OUTPUT_CSV_PATH}")
    if summary["missing_metrics"]:
        print("Missing metrics: " + ", ".join(summary["missing_metrics"]))
    else:
        print("All requested metrics are available.")


if __name__ == "__main__":
    main()
