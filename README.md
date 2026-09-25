````markdown
# Enhancing 6D Object Pose Estimation

## Project Overview

This project studies 6D object pose estimation using RGB, depth, and RGB-D inputs from the preprocessed LINEMOD dataset.

A 6D pose consists of:

- 3D translation: the object's position `(x, y, z)` relative to the camera;
- 3D rotation: the object's orientation, represented using a quaternion.

The repository contains:

- YOLO-based object localization;
- RGB-only pose estimation using a pretrained ResNet-50 backbone;
- depth-only pose estimation using a lightweight convolutional encoder;
- RGB-D pose estimation using RGB/depth feature fusion;
- quantitative ADD evaluation;
- translation and rotation error analysis;
- ADD threshold and ADD-0.1d evaluation;
- qualitative pose-prediction visualization; and
- an end-to-end YOLO-to-pose inference demonstration.

The main quantitative RGB, depth-only, and RGB-D pose results use ground-truth object bounding-box crops from the dataset loaders. YOLO-predicted bounding boxes are evaluated quantitatively on the full saved validation/test runs, and representative examples are plotted to show their effect on the pose-estimation pipeline.

The RGB-D architecture is a compact project-specific fusion model and is not a full reproduction of DenseFusion.

## Pipeline

```text
RGB / Depth Input
        |
        v
Object Localization
(Ground Truth or YOLO)
        |
        v
Object Crop
        |
        v
RGB / Depth / RGB-D
Pose Estimator
        |
        v
Translation + Quaternion Rotation
        |
        v
6D Pose
````

## Dataset

The project uses the preprocessed LINEMOD dataset located at:

```text
data/linemod/Linemod_preprocessed
```

The dataset contains:

* RGB images;
* 16-bit depth images;
* object bounding boxes;
* object masks;
* camera intrinsics;
* rotation and translation annotations; and
* PLY object models.

The reported experiments focus on LINEMOD object 1.

Pose crops are resized to `224 x 224`.

RGB inputs use ImageNet normalization. Depth crops are resized using nearest-neighbor interpolation, multiplied by the per-sample `depth_scale`, and divided by `1000` to convert the neural-network input to metres. The depth scale comes from each object's `info.yml` file and defaults to `1.0` when absent.

Translation targets, PLY object coordinates, pose errors, and ADD results are expressed in millimetres.

## Model Architecture

### Object Detection

The detection stage uses YOLO11n for object localization before pose estimation.

YOLO-predicted bounding boxes are compared with the corresponding ground-truth annotations. Representative examples show high overlap between the predicted and annotated object regions.

### RGB Pose Estimation

The RGB-only model uses a `torchvision` ImageNet-pretrained ResNet-50 backbone (`ResNet50_Weights.DEFAULT`).

The original classification layer is replaced by an identity mapping, producing a 2048-dimensional feature representation.

Two regression heads predict:

* a 3-dimensional translation vector;
* a 4-dimensional quaternion.

The predicted quaternion is normalized to unit length.

### Depth-Only Pose Estimation

The depth-only model receives a single-channel depth crop.

The depth encoder contains three convolutional layers:

```text
1 -> 32 -> 64 -> 128
```

Each convolution uses:

* kernel size `3`;
* stride `2`;
* padding `1`;
* ReLU activation;
* batch normalization.

Adaptive average pooling produces a 128-dimensional depth feature vector.

The fully connected layers are:

```text
128 -> 512 -> 256
```

with ReLU activations and dropout with probability `0.3`.

Separate heads predict:

```text
Translation: 256 -> 3
Rotation:    256 -> 4
```

The predicted quaternion is normalized to unit length.

### RGB-D Pose Estimation

The RGB-D model processes aligned RGB and depth crops using two separate feature-extraction branches.

The RGB branch uses a `torchvision` ImageNet-pretrained ResNet-50 (`ResNet50_Weights.IMAGENET1K_V2`) and produces a 2048-dimensional feature vector.

The depth branch uses the same lightweight convolutional encoder as the depth-only model and produces a 128-dimensional feature vector.

The two representations are concatenated:

```text
2048 + 128 = 2176 features
```

The fusion module is:

```text
2176 -> 512 -> 256
```

with ReLU activations and dropout with probability `0.3`.

Separate heads predict:

```text
Translation: 256 -> 3
Rotation:    256 -> 4
```

## Training

The three pose-estimation models are trained independently.

| Configuration           |                            RGB |                     Depth-only |                          RGB-D |
| ----------------------- | -----------------------------: | -----------------------------: | -----------------------------: |
| Object ID               |                              1 |                              1 |                              1 |
| Input size              |                      224 x 224 |                      224 x 224 |                      224 x 224 |
| Epochs                  |                             20 |                             20 |                             20 |
| Batch size              |                              4 |                              4 |                              4 |
| Learning rate           |                           1e-4 |                           1e-4 |                           1e-4 |
| Optimizer               |                           Adam |                           Adam |                           Adam |
| Validation fraction     |                            20% |                            20% |                            20% |
| Split seed              |                             42 |                             42 |                             42 |
| Rotation representation |                     Quaternion |                     Quaternion |                     Quaternion |
| Translation loss        |                            MSE |                            MSE |                            MSE |
| Rotation loss           | Sign-invariant quaternion loss | Sign-invariant quaternion loss | Sign-invariant quaternion loss |
| Checkpoint selection    |          Lowest validation ADD |          Lowest validation ADD |          Lowest validation ADD |

For the pose scripts, `num_workers=0`. Object 1 has 186 original training samples, so the default 20% split produces 149 training and 37 validation samples; the 1050-sample test split is untouched. Seed `42` controls only the train/validation split. Model initialization and shuffled DataLoader batches are not globally seeded. The RGB-D script exposes these values as command-line arguments; the RGB and depth-only scripts use the constants shown above.

YOLO11n is trained separately for object localization with 20 epochs, batch size 8, and image size 640. The YOLO training script does not explicitly set a random seed. Its prepared dataset has one class (`object_01`), maps the LineMOD train split to YOLO `train`, and maps the LineMOD test split to YOLO `val`.

## Training Objective

The pose-estimation models jointly optimize translation and rotation.

### Translation Loss

Translation is trained using mean squared error:

```text
L_t = MSE(t_pred, t_gt)
```

### Rotation Loss

Both predicted and ground-truth quaternions are normalized to unit length.

Because `q` and `-q` represent the same 3D rotation, the rotation loss is sign invariant:

```text
L_r = 1 - |q_pred^T q_gt|
```

The final training objective is:

```text
L = L_t + L_r
```

with equal translation and rotation weights.

The implementation uses `translation_weight = 1.0` and `rotation_weight = 1.0` for all three pose models.

## Evaluation Metrics

### ADD

Average Distance of Model Points (ADD) measures the mean Euclidean distance between corresponding 3D model points transformed using the predicted and ground-truth poses.

Lower ADD indicates more accurate pose estimation.

### Translation Error

Translation error measures the Euclidean distance between the predicted and ground-truth 3D translations:

```text
||t_pred - t_gt||_2
```

### Rotation Error

Rotation error measures the quaternion angular difference in degrees. After normalizing both quaternions, the implementation computes:

```text
2 x acos(clamp(|q_pred^T q_gt|, 0, 1))
```

### ADD-0.1d

A pose is considered successful when:

```text
ADD < 0.1 x object diameter
```

For object 1, the saved evaluation computes the diameter from `obj_01.ply`:

```text
Object diameter ≈ 102.10 mm
ADD-0.1d threshold ≈ 10.21 mm
```

## Results

### Mean ADD

| Model      |      Mean ADD |
| ---------- | ------------: |
| RGB-only   |     610.49 mm |
| Depth-only |     184.99 mm |
| RGB-D      | **134.74 mm** |

Relative to RGB-only:

* depth-only reduces mean ADD by approximately `69.7%`;
* RGB-D reduces mean ADD by approximately `77.9%`.

RGB-D also reduces mean ADD by approximately `27.2%` compared with the depth-only model.

### Translation Error

| Model      | Mean Translation Error |
| ---------- | ---------------------: |
| RGB-only   |              609.04 mm |
| Depth-only |              179.96 mm |
| RGB-D      |          **128.64 mm** |

The translation results closely follow the ADD results, indicating that the main improvement obtained from depth information is associated with more accurate estimation of the object's 3D position.

### Rotation Error

| Model      | Mean Rotation Error |
| ---------- | ------------------: |
| RGB-only   |          **73.43°** |
| Depth-only |              90.48° |
| RGB-D      |              90.33° |

Unlike translation performance, the RGB-only model obtains the lowest mean rotation error.

The overall ADD improvement achieved by RGB-D is therefore primarily associated with improved translation estimation rather than improved orientation estimation.

### ADD-0.1d Success Rate

| Model      | Successful / Total | Success Rate |
| ---------- | -----------------: | -----------: |
| RGB-only   |           0 / 1050 |         0.0% |
| Depth-only |           0 / 1050 |         0.0% |
| RGB-D      |           0 / 1050 |         0.0% |

Although depth-only and RGB-D substantially improve relative ADD performance, none of the evaluated models satisfies the strict ADD-0.1d criterion.

## YOLO Detection and End-to-End Pose Estimation

YOLO is used to replace ground-truth object localization in representative end-to-end examples.

The detector has two quantitative evaluation paths. `evaluate_yolo.py` reports validation mAP; the saved best recorded row has mAP@50 `0.99500` and mAP@50:95 `0.89686` at epoch 20. `evaluate_yolo_bbox_iou.py` evaluates all 1050 validation images by matching the highest-confidence class-0 prediction to the ground-truth box; the saved result has mean IoU `0.93875`, median IoU `0.94517`, and 99.90% of samples at IoU >= 0.5. Its figure shows three representative examples, not the complete quantitative evaluation.

`evaluate_yolo_pose_add.py` also evaluates the full 1050-sample test split, using both ground-truth and highest-confidence YOLO crops for the RGB and RGB-D pose models. The saved means are:

| Condition | Valid samples | Mean ADD |
| --------- | ------------: | --------: |
| RGB, ground-truth crop | 1050 | 610.45 mm |
| RGB, YOLO crop | 1050 | 611.64 mm |
| RGB-D, ground-truth crop | 1050 | 134.74 mm |
| RGB-D, YOLO crop | 1050 | 133.34 mm |

The accompanying figure selects three representative samples from that full run. `run_pose_pipeline.py` is a separate one-sample inference demonstration and is not a quantitative test-set evaluation.

Example pose comparisons are:

| Sample |   IoU | RGB GT ADD | RGB YOLO ADD | RGB-D GT ADD | RGB-D YOLO ADD |
| ------ | ----: | ---------: | -----------: | -----------: | -------------: |
| 0000   | 0.983 |   850.0 mm |     818.3 mm |     158.8 mm |       153.0 mm |
| 0623   | 0.966 |   465.2 mm |     472.6 mm |     127.5 mm |       131.1 mm |
| 1234   | 0.970 |   414.7 mm |     417.1 mm |      98.2 mm |        95.6 mm |

For the saved full test-set evaluation, replacing the ground-truth crop with the YOLO-predicted crop produces similar mean ADD for both pose models. The three rows shown above are representative examples selected by the plotting script, not the aggregate results.

## Visual Results

The project generates visualizations including:

* ADD threshold success curves;
* ADD error distributions;
* translation error comparisons;
* rotation error comparisons;
* YOLO-predicted versus ground-truth bounding boxes;
* the effect of YOLO crops on pose ADD; and
* qualitative pose-estimation examples.

Representative report figures include:

```text
add_threshold_curve.png
add_distribution_boxplot.png
rgb_depth_rgbd_add.png
translation_error_comparison.png
rotation_error_comparison.png
yolo_bbox_iou_examples.png
yolo_pose_add_comparison.png
```

Training-history figures are also written for each pose model: `rgb_training_*`, `depth_training_*`, and `rgbd_training_*`.

## Installation

From the project directory `6D-Object-Pose-Estimation` in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements.txt
```

The repository dependencies include PyTorch, torchvision, NumPy, Pillow, Matplotlib, PyYAML, and Ultralytics.

## Dataset and Weights

Place the preprocessed dataset at:

```text
data/linemod/Linemod_preprocessed
```

The repository uses the YOLO base checkpoint `yolo11n.pt`. Trained checkpoints are `checkpoints/best_model.pth` (RGB), `checkpoints/depth_best.pth` (depth-only), and `checkpoints/rgbd_best.pth` (RGB-D). The trained YOLO checkpoint used by evaluation is `runs/detect/runs/linemod_yolo/weights/best.pt`.

Machine-readable pose results are saved in `outputs/rgb_depth_rgbd_ablation.json`, `outputs/add_success_rate.json`, `outputs/add_threshold_curve.json`, `outputs/add_distribution.json`, `outputs/translation_rotation_errors.json`, and `outputs/yolo_pose_add_comparison.json`. `outputs/yolo_bbox_iou_metrics.json` stores the full validation IoU results.

Datasets, generated outputs, checkpoints, and large model weights should not normally be committed to version control.

## Usage

Run the following commands from the project directory after activating `.venv`.

### Explore the LINEMOD dataset

```powershell
python -m src.dataset_exploration
```

### Prepare the YOLO dataset

```powershell
python -m src.prepare_yolo_dataset
```

### Train and evaluate YOLO

```powershell
python train_yolo.py
python evaluate_yolo.py
```

### Train RGB, depth-only, and RGB-D pose models

```powershell
python train.py
python train_depth.py
python train_rgbd.py
```

### RGB pose inference

```powershell
python -m src.pose_inference
```

### RGB-D inference

```powershell
python infer_rgbd.py --sample-id 0
```

### RGB inference from a saved crop

`src.pose_inference` reads `outputs/crops/0000_crop.png`, so create that crop first with the crop script if it is not already present:

```powershell
python -m src.crop_detected_object
python -m src.pose_inference
```

To save YOLO prediction visualizations for the first ten validation images, use:

```powershell
python predict_yolo.py
```

### End-to-end YOLO localization followed by RGB pose estimation

```powershell
python run_pose_pipeline.py --mode rgb --sample-id 0
```

### End-to-end YOLO localization followed by RGB-D pose estimation

```powershell
python run_pose_pipeline.py --mode rgbd --sample-id 0
```

### Generate saved result summaries

```powershell
python generate_results_table.py
python plot_results.py
```

`generate_results_table.py` and `plot_results.py` summarize the legacy two-model `outputs/rgb_vs_rgbd.json` file; they do not generate the current depth-only ablation results. To reproduce the current three-modality result files and figures, use:

```powershell
python evaluate_rgb_depth_rgbd_ablation.py
python evaluate_add_success_rate.py
python evaluate_add_threshold_curve.py
python evaluate_add_distribution.py
python evaluate_translation_rotation_errors.py
python evaluate_yolo.py
python evaluate_yolo_bbox_iou.py
python evaluate_yolo_pose_add.py
```

Training can take a significant amount of time and should be started manually.

## Project Structure

```text
6D-Object-Pose-Estimation/
├── src/
│   ├── __init__.py
│   ├── checkpointing.py
│   ├── config.py
│   ├── linemod_dataset.py
│   ├── rgbd_dataset.py
│   ├── pose_model.py
│   ├── depth_pose_model.py
│   ├── rgbd_pose_model.py
│   ├── pose_loss.py
│   ├── metrics.py
│   ├── geometry.py
│   ├── model_loader.py
│   ├── paths.py
│   ├── pose_inference.py
│   ├── training.py
│   ├── depth_training.py
│   ├── rgbd_training.py
│   ├── training_history.py
│   ├── crop_detected_object.py
│   ├── visualize_yolo_bbox.py
│   ├── prepare_yolo_dataset.py
│   └── dataset_exploration.py
├── data/
│   └── linemod/
│       └── Linemod_preprocessed/
├── train_yolo.py
├── evaluate_yolo.py
├── evaluate_yolo_bbox_iou.py
├── evaluate_yolo_pose_add.py
├── train.py
├── train_depth.py
├── train_rgbd.py
├── compare_rgb_rgbd.py
├── evaluate_rgb_depth_rgbd_ablation.py
├── evaluate_add_success_rate.py
├── evaluate_add_threshold_curve.py
├── evaluate_add_distribution.py
├── evaluate_translation_rotation_errors.py
├── visualize_pose_predictions.py
├── predict_yolo.py
├── run_pose_pipeline.py
├── infer_rgbd.py
├── generate_results_table.py
├── plot_results.py
└── ../requirements.txt
```

The `outputs/`, `checkpoints/`, `runs/`, and dataset directories contain generated artifacts and downloaded data; their contents depend on the local experiment state.

## Reference Papers

The academic report discusses and cites established work on:

* LINEMOD;
* PoseCNN;
* DenseFusion;
* Pix2Pose;
* RGB-D attention-based fusion;
* surface-keypoint-based pose estimation;
* YOLO object detection; and
* ADD-based pose evaluation.

The complete verified bibliography is provided in the accompanying academic report.

## Limitations

* The reported experiments focus on LINEMOD object 1.
* Quantitative RGB, depth-only, and RGB-D comparisons use ground-truth object crops.
* YOLO-to-pose evaluation uses the saved full 1050-sample test run; the accompanying figure shows only representative examples.
* All three pose models obtain a 0% success rate under the strict ADD-0.1d criterion.
* Rotation estimation remains a major limitation of the current models.
* The RGB-D architecture is a simplified project-specific feature-fusion approach rather than a full reproduction of DenseFusion.
* The reported results do not establish performance across all LINEMOD objects or universal superiority of RGB-D.

## Report

The accompanying `report.tex` and `report_results.tex` are an older RGB-vs-RGB-D report summary. They do not document the current depth-only extension and their two-model saved ADD table differs from the current three-modality files above. The current README and machine-readable outputs are the source of truth for the extended experiment.