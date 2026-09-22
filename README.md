# Enhancing 6D Object Pose Estimation

This project uses the LineMOD dataset and PyTorch to compare RGB and RGB-D
6D pose estimation after YOLO object localization. LineMOD translations and
PLY model coordinates are in millimetres. RGB-D input depth is converted to
metres for the neural network while pose outputs and ADD remain in millimetres.

## Environment

From this directory, create or activate a virtual environment and install the
workspace requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements.txt
```

The dataset and pretrained weights are intentionally ignored by Git. Place the
preprocessed LineMOD data under `data/linemod/Linemod_preprocessed` and the
YOLO base weights at `yolo11n.pt`.

## Recommended execution order

```powershell
python -m src.dataset_exploration
python -m src.prepare_yolo_dataset
python train_yolo.py
python evaluate_yolo.py
python train.py
python train_rgbd.py
python compare_rgb_rgbd.py
```

Training commands are intentionally manual and can take a long time. The RGB
and RGB-D pose trainers select checkpoints using a validation slice of the
original training split, then evaluate the untouched LineMOD test split.

Useful inference commands:

```powershell
python src/pose_inference.py
python infer_rgbd.py --sample-id 0
python run_pose_pipeline.py --mode rgb --sample-id 0
python run_pose_pipeline.py --mode rgbd --sample-id 0
```

The YOLO validation command reports `mAP50` and `mAP50-95`; pose evaluation
reports mean ADD in millimetres and writes comparison results under `outputs`.