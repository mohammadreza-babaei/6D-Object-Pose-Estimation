# Enhancing 6D Object Pose Estimation

## Project Overview

This project studies 6D object pose estimation using RGB and RGB-D data from
the preprocessed LineMOD dataset. A 6D pose consists of:

- 3D translation: the object's position $(x,y,z)$ relative to the camera;
- 3D rotation: the object's orientation, represented here by a quaternion.

The repository contains:

- YOLO-based object localization;
- RGB-only pose estimation using a ResNet-50 feature extractor;
- RGB-D pose estimation with RGB/depth feature fusion;
- quantitative ADD evaluation;
- qualitative pose-prediction visualization; and
- an end-to-end YOLO-to-pose inference demonstration.

The quantitative RGB and RGB-D pose results use ground-truth object bounding-
box crops from the dataset loaders. They were not computed from YOLO-detected
crops. YOLO localization is evaluated separately using detection metrics.
The script `run_pose_pipeline.py` demonstrates the complete path in which YOLO
localizes an object and the resulting crop is passed to a pose estimator.

## Pipeline

```text
RGB image
		-> YOLO object localization
		-> object crop
		-> RGB or RGB-D pose estimator
		-> translation + quaternion rotation
		-> 6D pose
```

The RGB-D model combines RGB appearance features with depth information through
the two-branch fusion architecture implemented in `src/rgbd_pose_model.py`.
This is a compact project-specific fusion model and is not a full reproduction
of DenseFusion.

## Dataset

The project uses the preprocessed LineMOD dataset located at
`data/linemod/Linemod_preprocessed`. The repository contains RGB images,
16-bit depth images, object bounding boxes, object masks, camera information
and intrinsics, rotation and translation annotations, and PLY object models.

The reported experiments focus on object 1, whose original split contains 186
training samples and 1050 test samples. Pose crops are resized to `224 x 224`.
RGB inputs use ImageNet normalization. Depth is resized using the repository's
nearest-neighbour preprocessing and converted to metres for network input.
Translation targets, PLY object coordinates, pose errors, and ADD results
remain in millimetres.

## Model Architecture

### Object Detection

The detection stage uses the pretrained YOLO11n checkpoint configured by
`train_yolo.py`. It localizes the object before pose estimation in the
end-to-end inference demonstration.

### RGB Pose Estimation

`src/pose_model.py` uses an ImageNet-pretrained ResNet-50. Its classification
layer is replaced with an identity mapping. Two linear heads predict a
three-dimensional translation and a four-dimensional quaternion in `[w, x, y,
z]` order. The predicted quaternion is normalized to unit length.

### RGB-D Pose Estimation

`src/rgbd_pose_model.py` uses separate RGB and depth branches. The RGB branch
is an ImageNet-initialized ResNet-50 with its classification layer removed,
producing 2048 features. The depth branch is a lightweight CNN with channel
dimensions `1 -> 32 -> 64 -> 128`; each convolution uses stride 2 and is
followed by ReLU and batch normalization, then adaptive average pooling
produces 128 depth features.

The 2048 RGB features and 128 depth features are concatenated. The fusion
module maps 2176 features to 512, applies ReLU and dropout with probability
0.3, then maps to 256 features with another ReLU. Separate heads predict
translation and quaternion rotation, and the quaternion is normalized.

## Training

The verified training configurations are:

| Model | Epochs | Batch size | Other configuration |
| --- | ---: | ---: | --- |
| YOLO11n | 20 | 8 | Image size 640, seed 0 |
| RGB pose | 20 | 4 | Adam, learning rate `1e-4`, validation fraction 0.2, seed 42 |
| RGB-D pose | 20 | 4 | Learning rate `1e-4`, validation fraction 0.2, seed 42 |

The RGB and RGB-D pose trainers select checkpoints using validation ADD and
then evaluate the selected checkpoint on the untouched test split. Training is
manual and is not started by the reporting or visualization scripts.

## Evaluation Metrics

- **YOLO mAP@50 and mAP@50:95:** object-detection average precision at the
	indicated IoU thresholds.
- **ADD:** the mean Euclidean distance, in millimetres, between corresponding
	3D model points transformed by the predicted and ground-truth poses.
- **Translation loss:** mean squared error between predicted and target
	translations.
- **Quaternion rotation loss:** the sign-invariant loss
	$1 - |q_{pred}^T q_{target}|$, where $q$ and $-q$ represent the same
	rotation. The repository uses translation and rotation weights of `1.0`.

## Results

The following values are read from the saved repository results. Detection
metrics come from the best recorded YOLO `mAP50-95` row. Pose results are for
LineMOD object 1 and use ground-truth object crops.

| Evaluation | Metric | Result |
| --- | --- | ---: |
| YOLO | mAP@50 | 0.995 |
| YOLO | mAP@50:95 | 0.89686 |
| RGB pose | ADD | 589.5225 mm |
| RGB-D pose | ADD | 138.2051 mm |
| RGB to RGB-D | Relative ADD reduction | 76.5564% |

The RGB-D model has lower ADD in this experiment. This result is limited to
object 1 and the saved checkpoints; it does not establish universal RGB-D
superiority.

## Visual Results

The following figures exist locally in the repository:

![RGB versus RGB-D ADD comparison](outputs/rgb_vs_rgbd_add.png)

![Qualitative pose prediction examples](outputs/pose_prediction_examples.png)

The `outputs/` directory is ignored by the current `.gitignore`, so these
relative links will not display on GitHub unless selected figures are added to
version control separately. The ignore rules are intentionally unchanged.

## Installation

From the project directory `6D-Object-Pose-Estimation` in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements.txt
```

The repository's `requirements.txt` is located one directory above the project
directory. The listed dependencies include PyTorch, torchvision, NumPy, Pillow,
Matplotlib, PyYAML, and Ultralytics.

## Dataset and Weights

Place the preprocessed dataset at:

```text
data/linemod/Linemod_preprocessed
```

The YOLO preparation and training scripts use the object-1 dataset and the
repository's YOLO base checkpoint at `yolo11n.pt`. The repository also contains
an additional model file under `weights/`. Datasets, generated outputs,
checkpoints, and large model weights are excluded by the current Git ignore
rules and should not normally be committed.

## Usage

Run these commands from the project directory after activating `.venv`:

```powershell
# Explore the LineMOD structure and metadata
python -m src.dataset_exploration

# Prepare YOLO images and labels
python -m src.prepare_yolo_dataset

# Train and evaluate YOLO
python train_yolo.py
python evaluate_yolo.py

# Train RGB and RGB-D pose models
python train.py
python train_rgbd.py

# Compare saved RGB and RGB-D test results
python compare_rgb_rgbd.py

# Generate qualitative pose predictions
python visualize_pose_predictions.py
```

Useful inference commands are:

```powershell
# RGB pose inference on a saved crop
python -m src.pose_inference

# RGB-D inference for a LineMOD test sample
python infer_rgbd.py --sample-id 0

# End-to-end YOLO localization followed by RGB pose estimation
python run_pose_pipeline.py --mode rgb --sample-id 0

# End-to-end YOLO localization followed by RGB-D pose estimation
python run_pose_pipeline.py --mode rgbd --sample-id 0
```

Additional reporting utilities are available for saved outputs:

```powershell
python generate_results_table.py
python plot_results.py
```

The training commands can take a long time and should be run manually. Do not
run them merely to generate the README or report figures.

## Project Structure

```text
6D-Object-Pose-Estimation/
├── src/
│   ├── linemod_dataset.py
│   ├── rgbd_dataset.py
│   ├── pose_model.py
│   ├── rgbd_pose_model.py
│   ├── pose_loss.py
│   ├── metrics.py
│   ├── training.py
│   ├── rgbd_training.py
│   ├── prepare_yolo_dataset.py
│   ├── dataset_exploration.py
│   └── checkpointing.py
├── data/linemod/Linemod_preprocessed/
├── train_yolo.py
├── evaluate_yolo.py
├── train.py
├── train_rgbd.py
├── compare_rgb_rgbd.py
├── visualize_pose_predictions.py
├── run_pose_pipeline.py
├── infer_rgbd.py
├── generate_results_table.py
├── plot_results.py
├── report.tex
├── report_results.tex
└── ../requirements.txt
```

## Reference Papers

The project is motivated by established work on the LineMOD dataset and 6D
object pose estimation, RGB-based pose estimation, and RGB-D feature-fusion or
DenseFusion-style approaches. Verified paper titles, authors, venues, and
bibliographic metadata are not stored in this repository. The final academic
references will be added together with the Overleaf report after verification.

## Limitations

- The reported pose comparison focuses on LineMOD object 1.
- Quantitative ADD uses ground-truth object crops from the dataset loaders.
- Detector-crop ADD was not measured for the reported comparison.
- The RGB-D model is a simplified feature-fusion approach, not a full
	reproduction of DenseFusion.
- The saved results do not establish performance across all LineMOD objects or
	universal superiority of RGB-D over RGB.

## Report

The academic report is written in LaTeX in `report.tex` and includes the saved
results section through `\input{report_results}`. It is structured for
compilation in Overleaf using relative paths for figures and included TeX
content.