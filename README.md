# Machine Learning Based Adaptive Grip Force Control for Robotic Grippers

This project builds a supervised learning pipeline on top of an existing modular MuJoCo MJCF robotic arm and gripper design.

The robot definition is reused directly from:
- assets/main.xml
- assets/models/*.xml

No robot rebuild is performed. The implementation focuses on gripper control, grasp trial simulation, force/contact data extraction, CSV dataset generation, and supervised model training.

The control pipeline now includes:
- flexible lower-arm motion (pre-grasp -> approach -> lift)
- adaptive grip-force control using contact and normal-force feedback as tactile proxies

## Project Goals

1. Load existing robot from assets/main.xml.
2. Control arm and gripper (gripper-focused).
3. Execute grasping trials on cube/cylinder objects.
4. Extract force/contact and motion features.
5. Save dataset to data/raw/grip_dataset.csv.
6. Train supervised models for:
- grasp success prediction (classification)
- optimal grip command prediction (regression)

## Folder Structure

- assets/ : existing MuJoCo MJCF and mesh files
- simulation/ : simulation, object injection, control discovery, trial execution, feature extraction
- data/raw/ : generated datasets
- data/processed/ : optional processed feature sets
- models/artifacts/ : trained model artifacts and metrics
- scripts/ : CLI entry scripts

## Key Implementation Modules

- simulation/config.py : simulation and experiment configuration
- simulation/scene_builder.py : builds per-trial XML scene by injecting trial object into assets/main.xml
- simulation/robot_interface.py : discovers joint and actuator indices and provides control helpers
- simulation/data_extraction.py : extracts contacts and contact-force summaries from MuJoCo data
- simulation/trial_runner.py : runs one full grasp trial and returns structured trial record
- models/training_pipeline.py : training and evaluation for classification and regression models

## Environment Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

python -m pip install -r requirements.txt

## CLI Workflow

### 1) Inspect model controls and names

python scripts/inspect_model.py

### 2) Run one trial sanity check

python scripts/run_single_trial.py --object-type box --grip-command 6.0

### 3) Interactive viewer control

python scripts/view_simulation.py --object-type box --grip-command 6.0

Viewer console commands:
- open
- close
- grip X
- arm
- approach
- lift
- adapt on|off
- stats
- reset
- quit

### 4) Generate dataset

python scripts/generate_dataset.py --num-scenarios 40 --forces-per-scenario 10 --output data/raw/grip_dataset.csv --plot

### 5) Automatic grasp pose calibration sweep

python scripts/calibrate_grasp_pose.py --max-combos 120 --repeats 2 --write-config

Calibration behavior:
- sweeps arm_grasp_targets offsets
- sweeps object spawn offsets around nominal pose
- maximizes combined objective: lift success + object-gripper contact + height gain, penalizing slip
- writes tuned defaults back into simulation/config.py

### 6) Tactile-feedback plotting (standalone)

python scripts/plot_tactile_feedback.py --dataset data/raw/grip_dataset.csv --out-dir data/processed/plots

Generated plots:
- contact_force_over_trials.png
- success_vs_adaptive_final_grip_command.png
- slip_event_distribution.png

Dataset columns include:
- Features:
  - grip_command
  - adaptive_final_grip_command
  - gripper_joint_mean
  - object size/mass
  - object z-height features (initial/max/final/delta)
  - contact_count_total_peak
  - contact_count_obj_gripper_peak
  - mean_contact_normal_force
  - tactile_contact_mean
  - tactile_force_mean
  - slip_event_count
- Labels:
  - success (0/1)
  - optimal_grip_command (minimum successful adaptive command per scenario)

### 7) Train supervised models

python scripts/train_models.py --dataset data/raw/grip_dataset.csv --artifacts-dir models/artifacts

Saved artifacts:
- models/artifacts/logistic_success.joblib
- models/artifacts/random_forest_success.joblib
- models/artifacts/random_forest_optimal_force.joblib
- models/artifacts/torch_mlp_optimal_force.pt
- models/artifacts/torch_mlp_scaler.joblib
- models/artifacts/metrics.json

## Model Stack

Classification (grasp success):
- Baseline: Logistic Regression
- Advanced: Random Forest Classifier

Regression (optimal grip command):
- Advanced: Random Forest Regressor
- Neural model: PyTorch MLP Regressor

## Success Label Logic

A trial is labeled success (1) when either condition is met:
- Lift criterion: object z increase exceeds threshold
- Stable grasp criterion: sustained gripper-object contacts with limited displacement during hold phase

Otherwise the label is failure (0).

## Adaptive Grip Feedback Logic

During close and hold phases:
- If object-gripper contact count is below target, grip command is increased.
- If normal force is below target, grip command is increased.
- If normal force is too high, grip command is reduced.
- If slip-like behavior is detected, grip command is increased.

All updates are clipped to configured actuator command bounds.

## Notes

- Existing robot MJCF structure is preserved.
- Grasp objects are generated dynamically in Python for dataset control.
- Gripper control is the primary control signal.
- Arm movement is intentionally simple and only supports a repeatable pre-grasp setup.
