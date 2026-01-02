import multiprocessing
from scipy.stats import trim_mean
from SLM.SLM_Manager import SLM_Manager
import speech_to_text
import numpy as np
from text_viewer import TextViewer
from time import sleep
from tts import Speaker
import utils
import history as history
import argparse
import detector
import torch
import math
import faulthandler
import uart_transmitter
from enum import Enum
from computer_vision import ComputerVision
from camera import Camera
import pyzed.sl as sl
import cv2
import json
from utils import LABELS

faulthandler.enable()

def find_closest_object(new_position, object_dict, threshold):
        min_distance = float('inf')
        closest_obj_id = None

        for obj_id, positions in object_dict.items():
            if len(positions) > 0:
                last_position = positions[-1]
                distance = np.linalg.norm(new_position - last_position)
                if distance < min_distance and distance <= threshold:
                    min_distance = distance
                    closest_obj_id = obj_id
        
        return closest_obj_id

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    

    print("Press ENTER to begin recording")
    opt = parser.parse_args()
    cv = ComputerVision(opt)
    mgr = SLM_Manager(
    model_name='gpt-oss:120b', 
    mode="cloud", 
    api_key='',
    max_new_tokens=1024
    )
    mgr.load_model()
    print("Press ENTER to begin recording")
    input()
    
    stt_res = speech_to_text.transcribe_directly()
    print(f"STT Result: {stt_res}")
    
    cmd = stt_res


    cam = Camera()
    cam.open()
    detected_objects = []
    next_object_id = 0

    try:
        for _ in range(5):
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                frame = cam.get_frame()
                if frame is not None:
                    objects = cv.detect(frame)
                    
                    dict_objects = {}
                    for obj in objects:
                        # print(f"Object {obj.label}, Probability: {obj.probability}, BBox 2D: {obj.bounding_box_2d}\n")
                        label_name = LABELS[obj.label][1]
                        dict_objects[label_name] = dict_objects.get(label_name, 0) + 1
                            
                    detected_objects.append(dict_objects)    
    finally:
        cam.close()
        cv2.destroyAllWindows()
    
    print("\nFinal Detected Objects Summary:")
    
    if detected_objects:
        frames = []
        for i, obj in enumerate(detected_objects, start=1):
            frames.append(f"frame {i}: {obj}")
        frame_str = (
            "Objets détectés par la caméra : "
            + "; ".join(frames)
            + ". "
        )
    else:
        frame_str = "Aucun objet détecté par la caméra. "
        
    context_str = (
        "L'objet demandé par l'utilisateur peut ou non apparaître dans les frames. "
        "Si l'objet n'est pas présent, tu dois le signaler et demander des clarifications. "
    )

    cmd = (
        frame_str
        + context_str
        + "La commande de l'utilisateur est : "
        + cmd
    )
    print(cmd)


    # Feed the data to llama3.2 model
    result = mgr.generate_response(cmd)
    print("Raw result from model:\n", result)
    
    json_dump = json.loads(result)
    res = json_dump['response']    
    tts = Speaker(voice="Zira")
    tts.speak(res)

if __name__ == '__main__':
    main()
