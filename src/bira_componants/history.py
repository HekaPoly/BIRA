from dataclasses import dataclass
import json
import sys
import numpy as np
import ast
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cv_viewer import labels

root = Path(__file__).resolve().parents[1]
target = root / "logs"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = target / f"{timestamp}_history.txt"

@dataclass
class ObjectsOutput:
    nObjects: str                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    objects: list

@dataclass
class ObjectOutput:
    label : str
    position: list # [x, y, z]
    dimensions: list # [width, height, length]

def write_json(obj_output) :
    target.mkdir(parents = True, exist_ok = True)
    with open(filename, "a") as f:
        f.write(str(obj_output.__dict__))
        f.write('\n')
        f.close()

def write_history(objects) :
    objects_out = []

    for obj in objects:
        if len(obj.bounding_box) == 0 : continue  
        if np.isnan(obj.position).any(): continue
     
        position = list(obj.position)
        label = labels.labelDict[int(obj.raw_label)] 
        dimensions = list(obj.dimensions)
        
        obj_output = ObjectOutput(label=label, position=position, dimensions=dimensions) 
        objects_out.append(obj_output)  
    
    objs_output = ObjectsOutput(nObjects=len(objects), objects=objects_out) 
    write_json(objs_output)

def get_distance(z_values):
    first_average = sum(z_values) / len(z_values)
    true_z_values = []
    for val in z_values:
        if (val < first_average + 0.3) and (val > first_average - 0.3):
            true_z_values.append(val)
        
    if len(true_z_values) == 0:
        return

    true_average = sum(true_z_values) / len(true_z_values)
    return true_average
