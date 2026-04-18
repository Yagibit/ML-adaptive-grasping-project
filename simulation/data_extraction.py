from __future__ import annotations

import mujoco
import numpy as np


def extract_contact_stats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trial_object_geom_id: int,
    gripper_geom_ids: list[int],
) -> dict[str, float]:
    total_contacts = int(data.ncon)
    gripper_geom_set = set(gripper_geom_ids)

    obj_gripper_contacts = 0
    normal_forces: list[float] = []

    for i in range(total_contacts):
        contact = data.contact[i]
        geom_pair = {int(contact.geom1), int(contact.geom2)}

        if trial_object_geom_id in geom_pair and any(g in geom_pair for g in gripper_geom_set):
            obj_gripper_contacts += 1
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, i, force)
            normal_forces.append(float(abs(force[0])))

    return {
        "contact_count_total": float(total_contacts),
        "contact_count_obj_gripper": float(obj_gripper_contacts),
        "mean_contact_normal_force": float(np.mean(normal_forces)) if normal_forces else 0.0,
    }
