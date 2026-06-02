import numpy as np
from pymol import cmd
import logging
logger = logging.getLogger(__name__)

def extract_point_cloud(obj_name: str) -> np.ndarray:
    model = cmd.get_model(obj_name)
    if not model.atom:
        return np.empty((0, 3))
    return np.array([atom.coord for atom in model.atom], dtype=float)

def create_pymol_cloud(coords: np.ndarray, name: str, scores: np.ndarray = None):
    if coords is None or len(coords) == 0:
        logger.warning(f"create_pymol_cloud: empty coordinates for {name}")
        return
    if name in cmd.get_object_list():
        cmd.delete(name)

    for i, (x, y, z) in enumerate(coords):
        b = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        cmd.pseudoatom(object=name, pos=[x, y, z], name=f"P{i}", resn="SIT", b=b)

    cmd.show("spheres", name)
    cmd.set("sphere_scale", 0.8, name)
    cmd.set("sphere_quality", 2, name)           
    cmd.set("sphere_transparency", 0.4, name)
    
    cmd.set("ambient", 0.2, name)                
    cmd.set("direct", 0.8, name)
    cmd.set("specular", 0.5, name)

    if scores is not None and len(scores) > 0:
        cmd.spectrum("b", "rainbow", name, minimum=min(scores), maximum=max(scores))
    else:
        cmd.color("cyan", name)                

    logger.info(f"Created cloud {name} with {len(coords)} points")