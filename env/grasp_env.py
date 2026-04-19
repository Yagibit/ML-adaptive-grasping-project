from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from env.config import EnvConfig, default_env_config


class GraspEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str = "none") -> None:
        super().__init__()
        self.config = config or default_env_config()
        self.render_mode = render_mode

        self.model = self._load_model_with_runtime_fix(self.config.model_path)
        self.data = mujoco.MjData(self.model)

        # Contact and solver stabilization for grasping.
        self.model.geom_friction[:] = np.array([1.5, 0.005, 0.0001], dtype=float)
        self.model.opt.iterations = 100
        self.model.opt.ls_iterations = 50

        self._cube_body = self._body_id("cube")
        self._ee_body = self._first_body_id(list(self.config.ee_body_candidates))
        self._cube_qpos_adr = self._cube_free_joint_qpos_adr()

        self._actuator_ids = [self._actuator_id(name) for name in self.config.actuator_names]
        self.action_low = self.model.actuator_ctrlrange[self._actuator_ids, 0].astype(np.float32)
        self.action_high = self.model.actuator_ctrlrange[self._actuator_ids, 1].astype(np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        self._obs_dim = self._build_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self.last_action = np.zeros(len(self._actuator_ids), dtype=np.float32)
        self.step_count = 0

    def _load_model_with_runtime_fix(self, model_path: Path) -> mujoco.MjModel:
        try:
            return mujoco.MjModel.from_xml_path(str(model_path))
        except ValueError as exc:
            msg = str(exc)
            if "repeated name 'gripper_ctrl' in actuator" not in msg:
                raise

            tree = ET.parse(model_path)
            root = tree.getroot()
            actuator = root.find("actuator")
            if actuator is None:
                raise

            removed = False
            for child in list(actuator):
                if child.tag == "position" and child.get("name") == "gripper_ctrl":
                    actuator.remove(child)
                    removed = True
                    break
            if not removed:
                raise

            fixed_path = model_path.parent / "main.runtime.fixed.xml"
            tree.write(fixed_path, encoding="ascii", xml_declaration=True)
            return mujoco.MjModel.from_xml_path(str(fixed_path))

    def _actuator_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0 and name == "gripper_ctrl":
            for candidate in ["gripper_ctrl", "gripper_ctrl", "ctrl", "gripper_ctrl"]:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
                if idx >= 0:
                    break
        if idx < 0:
            raise ValueError(f"Actuator not found: {name}")
        return int(idx)

    def _body_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"Body not found: {name}")
        return int(idx)

    def _first_body_id(self, candidates: list[str]) -> int:
        for name in candidates:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                return int(idx)
        raise ValueError(f"None of the body names were found: {candidates}")

    def _cube_free_joint_qpos_adr(self) -> int:
        jnt_count = int(self.model.body_jntnum[self._cube_body])
        jnt_adr = int(self.model.body_jntadr[self._cube_body])
        for j in range(jnt_adr, jnt_adr + jnt_count):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[j])
        return -1

    def _set_cube_pose(self, x: float, y: float, z: float) -> None:
        if self._cube_qpos_adr < 0:
            return
        q = self._cube_qpos_adr
        self.data.qpos[q + 0] = x
        self.data.qpos[q + 1] = y
        self.data.qpos[q + 2] = z
        self.data.qpos[q + 3] = 1.0
        self.data.qpos[q + 4] = 0.0
        self.data.qpos[q + 5] = 0.0
        self.data.qpos[q + 6] = 0.0
        self.data.qvel[q : q + 6] = 0.0

    def get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body].copy()

    def get_cube_vel(self) -> np.ndarray:
        return self.data.cvel[self._cube_body, 3:6].copy()

    def get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee_body].copy()

    def contact_count(self) -> int:
        return int(self.data.ncon)

    def _joint_features(self) -> tuple[np.ndarray, np.ndarray]:
        qpos_vals = []
        qvel_vals = []
        for act_id in self._actuator_ids:
            j_id = int(self.model.actuator_trnid[act_id, 0])
            qadr = int(self.model.jnt_qposadr[j_id])
            vadr = int(self.model.jnt_dofadr[j_id])
            qpos_vals.append(float(self.data.qpos[qadr]))
            qvel_vals.append(float(self.data.qvel[vadr]))
        return np.array(qpos_vals, dtype=np.float32), np.array(qvel_vals, dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        joint_qpos, joint_qvel = self._joint_features()
        cube_pos = self.get_cube_pos().astype(np.float32)
        cube_vel = self.get_cube_vel().astype(np.float32)
        ee_pos = self.get_ee_pos().astype(np.float32)
        return np.concatenate([joint_qpos, joint_qvel, cube_pos, cube_vel, ee_pos], axis=0).astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, bool]:
        cube_pos = self.get_cube_pos()
        cube_vel = self.get_cube_vel()
        ee_pos = self.get_ee_pos()

        dist = float(np.linalg.norm(cube_pos - ee_pos))
        contact_bonus = 0.02 * min(10.0, float(self.data.ncon))
        smooth_penalty = 0.01 * float(np.linalg.norm(action - self.last_action))

        stable_grasp = int(self.data.ncon) >= 2 and float(np.linalg.norm(cube_vel)) < 0.15
        stable_bonus = 0.5 if stable_grasp else 0.0

        success = cube_pos[2] > self.config.lift_success_z
        lift_bonus = 1.0 if success else 0.0

        reward = -dist + contact_bonus + stable_bonus + lift_bonus - smooth_penalty
        return float(reward), bool(success)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # Neutral controls and deterministic cube pose.
        neutral = np.array([0.3, 0.3, 0.45, 0.55, 0.1, 0.8], dtype=np.float32)
        neutral = np.clip(neutral, self.action_low, self.action_high)
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = float(neutral[i])

        tx, ty, tz = self.config.target_cube_pos
        self._set_cube_pose(tx, ty, tz)
        mujoco.mj_forward(self.model, self.data)

        self.last_action = neutral.copy()
        obs = self._build_obs()
        info = {"success": False}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_low, self.action_high)

        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = float(action[i])

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Keep cube reachable if numerical instability sinks it.
        if self._cube_qpos_adr >= 0 and float(self.data.qpos[self._cube_qpos_adr + 2]) < 0.06:
            tx, ty, _ = self.config.target_cube_pos
            self._set_cube_pose(tx, ty, 0.08)
            mujoco.mj_forward(self.model, self.data)

        reward, success = self._reward(action)
        obs = self._build_obs()
        truncated = self.step_count >= self.config.max_steps
        terminated = success
        info = {
            "success": success,
            "cube_z": float(self.get_cube_pos()[2]),
            "contacts": int(self.data.ncon),
            "ee_pos": self.get_ee_pos(),
        }
        self.last_action = action.copy()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
