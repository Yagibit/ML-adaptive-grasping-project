# Machine Learning Based Adaptive Grip Force Control for Robotic Grippers

## Project Structure

adaptive-grip-force-ml/
- assets/
  - meshes/ (unchanged)
  - models/ (unchanged)
  - main.xml (edited as requested)
- scripts/
  - run_simulation.py
  - generate_dataset.py
  - train_model.py
  - evaluate_model.py
- data/
  - grip_dataset.csv
- models/
  - trained_model.pkl
- requirements.txt
- README.md

## Rules Implemented

- Robot loads from assets/main.xml
- Robot modular structure is preserved
- Simple cube object is in assets/main.xml
- Controls are normalized in [0, 1]
- Actuators in assets/main.xml use ctrlrange="0 1"

## Setup

python -m pip install -r requirements.txt

## Run Simulation

python scripts/run_simulation.py

## Generate Dataset

python scripts/generate_dataset.py --trials 300 --output data/grip_dataset.csv

Collected columns:
- wrist_ctrl
- hand_ctrl
- contact_count
- object_height
- success

Label rule:
- success = 1 if object height increases
- success = 0 otherwise

## Train Model

python scripts/train_model.py --dataset data/grip_dataset.csv --output models/trained_model.pkl

Model:
- RandomForestClassifier

Features:
- wrist_ctrl
- hand_ctrl
- contact_count

Target:
- success

## Evaluate Model

python scripts/evaluate_model.py --dataset data/grip_dataset.csv --model models/trained_model.pkl --out-dir models

Printed metrics:
- Accuracy
- Precision
- Recall
- F1-score

Saved plots:
- confusion_matrix.png
- wrist_ctrl_vs_success.png
- contact_count_vs_success.png

## Hybrid Imitation + RL Pipeline

New research-oriented modules are provided:
- env/
  - config.py
  - grasp_env.py
  - expert_policy.py
- data/
  - collect_expert_data.py
  - trajectory_dataset.py
- models/
  - policy_network.py
- train_supervised/
  - train_bc.py
- train_rl/
  - train_rl.py
- scripts/
  - evaluate_grasp_learning.py

### Stage 1: Expert Data Generation

python data/collect_expert_data.py --episodes 120 --out-npz data/datasets/expert_trajectories.npz --save-h5

Outputs:
- states s_t
- actions a_t
- next_states s_{t+1}
- rewards, dones

### Stage 2: Supervised Behavior Cloning

python train_supervised/train_bc.py --dataset data/datasets/expert_trajectories.npz --out models/checkpoints/bc_policy.pt

### Stage 3: RL Refinement (PPO/SAC)

python train_rl/train_rl.py --algo ppo --timesteps 200000 --bc-path models/checkpoints/bc_policy.pt --out models/rl/rl_policy.zip

### Evaluation (Success Rate)

python scripts/evaluate_grasp_learning.py --mode expert
python scripts/evaluate_grasp_learning.py --mode bc --bc-path models/checkpoints/bc_policy.pt
python scripts/evaluate_grasp_learning.py --mode rl --algo ppo --rl-path models/rl/rl_policy.zip
