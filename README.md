# ML-Adaptive-Grasping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-red.svg)](https://mujoco.org/)

This repository implements a **Supervised Learning (Behavior Cloning)** pipeline for robotic grasping. Utilizing a modular MuJoCo-based environment, the system learns to map environment states to actuator commands by mimicking scripted expert trajectories.

---

## 📌 Overview

The project provides an end-to-end workflow to approximate robotic control without the computational overhead of Reinforcement Learning. 

### Key Features
* **Expert Mimicry:** Uses a scripted controller to generate high-fidelity grasping data.
* **Behavior Cloning (BC):** Trains a Deep Neural Network to map state-action pairs $(S \to A)$.
* **MuJoCo Integration:** Leverages high-speed physics simulation for data collection and testing.
* **Modular Architecture:** Separate modules for environment physics, data ingestion, and model training.

---

## 📂 Project Structure

```text
adaptive-grip-force-ml/
├── assets/             # MJCF models (.xml), meshes, and reference images
├── env/                # MuJoCo environment wrappers and expert policy logic
├── data/               # Data collection scripts and trajectory storage
├── models/             # Saved PyTorch/Pickle model checkpoints
├── scripts/            # Simulation entry points and utility tools
├── train_supervised/   # Core Behavior Cloning training logic
├── requirements.txt    # Dependency list
└── README.md           # Documentation
```

---

## 🛠️ Installation

### Prerequisites
* Python 3.8 or higher
* MuJoCo physics engine

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/adaptive-grip-force-ml.git
cd adaptive-grip-force-ml

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

### 1. Visualize the Environment
Before training, verify the simulation and expert motion in the MuJoCo viewer:
```bash
python scripts/run_simulation.py
```

### 2. Data Collection
Generate the expert trajectory dataset. This records joint angles, object coordinates, and expert actuator commands:
```bash
python data/collect_expert_data.py \
    --episodes 120 \
    --out-npz data/datasets/expert_trajectories.npz \
    --save-h5
```

### 3. Policy Training
Train the Behavior Cloning model. This script optimizes a network to predict the expert's next move based on current state observations:
```bash
python train_supervised/train_bc.py \
    --dataset data/datasets/expert_trajectories.npz \
    --out models/checkpoints/bc_policy.pt
```

### 4. Baseline Evaluation (Optional)
A legacy classifier is included for performance benchmarking:
```bash
python scripts/train_model.py \
    --dataset data/grip_dataset.csv \
    --output models/trained_model.pkl
```

---

## 📊 Evaluation Metrics

The performance of the learned policy is quantified by:

| Metric | Description |
| :--- | :--- |
| **Grasp Success Rate** | Percentage of trials where the cube is successfully lifted. |
| **Stability** | Maintenance of grip force during the duration of the lift. |
| **Reproducibility** | Variance of success across different initial object placements. |

---

## ⚠️ Notes & Limitations

* **Expert Dependence:** The model's upper-bound performance is strictly limited by the quality of the scripted expert.
* **Offline Learning:** No online adaptation is performed; the policy does not learn from its own mistakes (standard BC limitation).
* **Simulation vs. Reality:** Physics are tuned for stability; "Sim-to-Real" transfer may require additional domain randomization.
