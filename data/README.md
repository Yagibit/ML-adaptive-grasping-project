# data

Dataset generation and trajectory storage code.

- collect_expert_data.py records demonstrations from scripted control.
- trajectory_dataset.py provides PyTorch dataset loading for trajectory files.
- datasets/: generated npz/h5 expert trajectories.
- grip_dataset.csv: legacy tabular dataset used by classical baseline scripts.