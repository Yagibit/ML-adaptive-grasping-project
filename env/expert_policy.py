from __future__ import annotations

import numpy as np

from env.grasp_env import GraspEnv


def _toward(current: float, target: float, step: float) -> float:
    if abs(target - current) <= step:
        return target
    if target > current:
        return current + step
    return current - step


class ScriptedExpertPolicy:
    """Simple staged heuristic: approach -> align -> grasp -> lift."""

    def __init__(self, env: GraspEnv) -> None:
        self.env = env
        self.stage = "approach"
        self.align_count = 0
        self.grasp_count = 0
        self.lift_count = 0
        self.contact_count = 0

    def reset(self) -> None:
        self.stage = "approach"
        self.align_count = 0
        self.grasp_count = 0
        self.lift_count = 0
        self.contact_count = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        cube = self.env.get_cube_pos()
        ee = self.env.get_ee_pos()
        ex = float(cube[0] - ee[0])
        ey = float(cube[1] - ee[1])
        ez = float((cube[2] + 0.04) - ee[2])

        action = self.env.last_action.copy()

        if self.stage == "approach":
            action[0] = _toward(action[0], 0.78, 0.004)
            action[1] = _toward(action[1], 0.84, 0.004)
            action[2] = _toward(action[2], 0.40, 0.003)
            action[3] = _toward(action[3], 0.30, 0.003)
            action[4] = _toward(action[4], 0.10, 0.003)
            action[5] = _toward(action[5], 0.80, 0.01)

            if abs(ex) < 0.045 and abs(ey) < 0.02 and abs(ez) < 0.055:
                self.stage = "align"

        elif self.stage == "align":
            action[0] = _toward(action[0], 0.75, 0.002)
            action[1] = _toward(action[1], 0.82, 0.002)
            action[2] = _toward(action[2], 0.42, 0.002)
            action[3] = _toward(action[3], 0.16, 0.002)
            action[4] = _toward(action[4], 0.10, 0.003)
            action[5] = _toward(action[5], 0.80, 0.01)

            self.align_count += 1
            if self.align_count > 90:
                self.stage = "grasp"

        elif self.stage == "grasp":
            action[0] = _toward(action[0], 0.75, 0.001)
            action[1] = _toward(action[1], 0.82, 0.001)
            action[2] = _toward(action[2], 0.42, 0.001)
            action[3] = _toward(action[3], 0.16, 0.001)
            action[5] = _toward(action[5], -0.95, 0.012)

            if self.env.contact_count() > 0:
                self.contact_count += 1

            self.grasp_count += 1
            if self.grasp_count > 140 or self.contact_count > 20:
                self.stage = "lift"

        else:  # lift
            action[0] = _toward(action[0], 0.68, 0.002)
            action[1] = _toward(action[1], 0.70, 0.002)
            action[2] = _toward(action[2], 0.36, 0.002)
            action[3] = _toward(action[3], 0.20, 0.001)
            action[5] = _toward(action[5], -0.95, 0.01)

            self.lift_count += 1
            if self.lift_count > 180:
                self.reset()

        return np.clip(action, self.env.action_low, self.env.action_high)
