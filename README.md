# Supervised MuJoCo Grasping Demo

This repository demonstrates a **Supervised Learning (Behavior Cloning)** approach for robotic grasping using a modular MuJoCo-based environment. Rather than relying on Reinforcement Learning, the system learns a control policy directly from expert demonstrations.

---

## 📌 Overview

The project implements a straightforward machine learning pipeline to approximate robotic control:

- **Expert Demonstration:** A scripted controller executes a precise grasping motion  
- **Data Logging:** State-action pairs (joint angles, object coordinates, and actuator commands) are recorded  
- **Behavior Cloning:** A neural network is trained to map the environment state to the expert's actions  
- **Evaluation:** The trained model is tested on its ability to lift and hold the target object  

---

## 📂 Project Structure


adaptive-grip-force-ml/
├── assets/ # MJCF models, mesh files, and reference images
├── env/ # Simulation environment and scripted expert policy
├── data/ # Data collection scripts and generated datasets
├── models/ # Saved model weights and checkpoints
├── scripts/ # Entry-point scripts for simulation and utilities
├── train_supervised/ # Behavior cloning (SL) training logic
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🛠️ Installation

Ensure you have Python 3.8 or higher installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/adaptive-grip-force-ml.git
cd adaptive-grip-force-ml

# Install required packages
pip install -r requirements.txt
🚀 Execution Workflow
1. Run the Simulation

Visualize the environment and the expert's grasping motion in the MuJoCo viewer:

python scripts/run_simulation.py
2. Collect Expert Data

Generate a dataset of expert trajectories. This records the state-action pairs necessary for training:

python data/collect_expert_data.py \
    --episodes 120 \
    --out-npz data/datasets/expert_trajectories.npz \
    --save-h5
3. Train the Model

Train the Behavior Cloning policy. The model learns to map:

(state) → (action)
python train_supervised/train_bc.py \
    --dataset data/datasets/expert_trajectories.npz \
    --out models/checkpoints/bc_policy.pt
4. Legacy Classifier (Optional)

A simpler classification-based model is included for baseline comparison:

python scripts/train_model.py \
    --dataset data/grip_dataset.csv \
    --output models/trained_model.pkl
📊 Evaluation & Metrics

The system evaluates the learned policy based on:

Grasp Success: Whether the robot successfully lifts the cube
Stability: Whether the grip remains firm during lifting
Reproducibility: Consistency across multiple trials
⚠️ Notes & Limitations
Behavior Cloning: Performance depends on the quality of the scripted expert
No Online Adaptation: This project focuses strictly on supervised learning (no reinforcement learning)
Simplified Physics: Contact dynamics are tuned for stability in simulation
Visual Target: The system aims to replicate the configuration shown in robot_grip.jpg

---
