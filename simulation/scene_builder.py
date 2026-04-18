from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from simulation.config import ObjectSpec


def _format_vec3(values: tuple[float, float, float]) -> str:
    return " ".join(f"{v:.6f}" for v in values)


def build_scene_xml(base_model_path: Path, output_path: Path, obj: ObjectSpec) -> Path:
    """Create a temporary scene by adding a trial object body to the existing MJCF."""
    tree = ET.parse(base_model_path)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")

    worldbodies = root.findall("worldbody")
    if not worldbodies:
        worldbody = ET.SubElement(root, "worldbody")
    else:
        worldbody = worldbodies[-1]

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "trial_object",
            "pos": _format_vec3(obj.spawn_pos_xyz),
        },
    )
    ET.SubElement(body, "joint", {"name": "trial_object_freejoint", "type": "free"})

    if obj.object_type == "cylinder":
        size_value = f"{obj.size_xyz[0]:.6f} {obj.size_xyz[2]:.6f}"
    else:
        size_value = _format_vec3(obj.size_xyz)

    geom_attrib = {
        "name": "trial_object_geom",
        "type": obj.object_type,
        "mass": f"{obj.mass:.6f}",
        "size": size_value,
        "rgba": "0.1 0.8 0.2 1",
        "friction": "0.9 0.05 0.02",
        "solimp": "0.95 0.995 0.001",
        "solref": "0.002 1",
    }
    ET.SubElement(body, "geom", geom_attrib)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="ascii", xml_declaration=True)
    return output_path
