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
    
    parser.add_argument('--cv', type=str, default=None, help='Showcase cv abilities of BIRA for specified duration (use inf for infinity)')
    parser.add_argument('--stt', action="store_true", help='Run speech to text app')
    parser.add_argument('--motors', help='Testing motors app')

    opt = parser.parse_args()
    # input("Press ENTER to begin recording")

    cv = ComputerVision(opt)
    # stt_res = speech_to_text.transcribe_directly()
    # print(f"STT Result: {stt_res}")

    cam = Camera()
    cam.open()
    detected_objects = {}
    next_object_id = 0
    MAX_DISTANCE: float = 7.0
    PROXIMITY_THRESHOLD: float = 0.3

    try:
        for _ in range(3):
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                frame = cam.get_frame()
                if frame is not None:
                    objects = cv.detect(frame)
                    print(f"Detected {len(objects)} objects")
                    
                    
                    for obj in objects:
                        print(f"Object {obj.label}, Probability: {obj.probability}, BBox 2D: {obj.bounding_box_2d}\n")
                        bbox = obj.bounding_box_2d
                        x_center = int((bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4)
                        y_center = int((bbox[0][1] + bbox[1][1] + bbox[2][1] + bbox[3][1]) / 4)
                        current_position = np.array([x_center, y_center])
                        objects_dict = detected_objects.setdefault(obj.label , {})
                        
                        closest_id = find_closest_object(current_position, objects_dict, PROXIMITY_THRESHOLD)
                        if closest_id is not None:
                            # Append the position to the existing object's history
                            objects_dict[closest_id] = np.vstack([objects_dict[closest_id], current_position])
                        else:
                            # Create a new object with a unique ID and initialize its history
                            obj_id = next_object_id
                            next_object_id += 1
                            objects_dict[obj_id] = np.array([current_position])
                            
                    cv2.imshow("BIRA Camera View", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            else:
                print("Press ESC to stop detection...", end='\r')
    finally:
        cam.close()
        cv2.destroyAllWindows()
    
    print(f"\nDetected {len(detected_objects)} objects:")
    print(detected_objects)
    # print(detected_objects)
    # print(stt_res)

    # Feed the data to llama3.2 model
    # cmd = stt_res
    # mgr = SLM_Manager(model_name='BIRA')
    # mgr.load_model()

    # if not cmd:
    #     print('Exemple : python SLM_Manager.py "BIRA, donne moi la pomme bleu"')

    # extraction = mgr.analyze_command(cmd)
    # json_dump = json.dumps(extraction.to_payload(), ensure_ascii=False)
    # res = json_dump['response']
    
    # print(f'JSON Response: {json_dump}')
    # print(f"status={extraction.status} confidence={extraction.confidence}")
    
    res = "Voici les objets que j'ai détectés : " + ", ".join([LABELS[obj.label][1] for obj in detected_objects])
    print(res)
    tts = Speaker(voice="Zira")
    tts.speak(res)

if __name__ == '__main__':
    main()
