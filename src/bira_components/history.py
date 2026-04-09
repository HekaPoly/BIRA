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
session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = target / f"{session_timestamp}_history.txt"
conversation_filename = target / f"{session_timestamp}_conversation.jsonl"
events_filename = target / f"{session_timestamp}_events.jsonl"

@dataclass
class ObjectsOutput:
    nObjects: str                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    objects: list

@dataclass
class ObjectOutput:
    label : str
    position: list # [x, y, z]
    dimensions: list # [width, height, length]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_json_line(path: Path, payload: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
        f.write("\n")


def log_conversation(role: str, content: str, **details) -> None:
    entry = {
        "timestamp": _timestamp(),
        "role": str(role),
        "content": "" if content is None else str(content),
    }
    if details:
        entry["details"] = details
    _append_json_line(conversation_filename, entry)


def log_event(event: str, **details) -> None:
    entry = {
        "timestamp": _timestamp(),
        "event": str(event),
    }
    if details:
        entry["details"] = details
    _append_json_line(events_filename, entry)

def write_json(obj_output) :
    target.mkdir(parents = True, exist_ok = True)
    with open(filename, "a", encoding="utf-8") as f:
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
