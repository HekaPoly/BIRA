import multiprocessing
# from scipy.stats import trim_mean
from SLM.SLM_Manager import SLM_Manager
# import speech_to_text
import numpy as np
from text_viewer import TextViewer
from time import sleep
from tts import Speaker
import utils
import history as history
import argparse
# import detector
# import torch
import math
import faulthandler
# import uart_transmitter
from enum import Enum
# from computer_vision import ComputerVision
# from camera import Camera
# import pyzed.sl as sl
# import cv2
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    # parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    # parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    

    # opt = parser.parse_args()
    # input("Press ENTER to begin recording")

    # cv = ComputerVision(opt)
    # stt_res = speech_to_text.transcribe_directly()
    # print(f"STT Result: {stt_res}")

    # cam = Camera()
    # cam.open()
    # detected_objects = []
    # next_object_id = 0
    # MAX_DISTANCE: float = 7.0
    # PROXIMITY_THRESHOLD: float = 0.3

    # try:
    #     for _ in range(5):
    #         if cam.grab() == sl.ERROR_CODE.SUCCESS:
    #             frame = cam.get_frame()
    #             if frame is not None:
    #                 objects = cv.detect(frame)
                    
    #                 dict_objects = {}
    #                 for obj in objects:
    #                     # print(f"Object {obj.label}, Probability: {obj.probability}, BBox 2D: {obj.bounding_box_2d}\n")
    #                     label_name = LABELS[obj.label][1]
    #                     dict_objects[label_name] = dict_objects.get(label_name, 0) + 1
                            
    #                 detected_objects.append(dict_objects)    
    # finally:
    #     cam.close()
    #     cv2.destroyAllWindows()
    
    # print("\nFinal Detected Objects Summary:")
    # for i, obj in enumerate(detected_objects):
    #         print(f"Frame {i+1}: {obj}")

    # print(detected_objects)
    # print(stt_res)

    # Feed the data to llama3.2 model
    cmd = "BIRA, donne moi la pomme bleu"
    mgr = SLM_Manager(
        model_name='gpt-oss:120b', 
        mode="cloud", 
        api_key=':)',
        max_new_tokens=512
        )
    mgr.load_model()

    # if not cmd:
    #     print('Exemple : python SLM_Manager.py "BIRA, donne moi la pomme bleu"')

    result = mgr.generate_response(cmd)
    print("Raw result from model:\n", result)
    json_dump = json.loads(result)
    
    res = json_dump['response']
    
    print(f'\nresponse: \n{res}')
    print(f"status={json_dump['status']} confidence={json_dump['confidence']}")
    
    # res = "Voici les objets que j'ai détectés : " + ", ".join([LABELS[obj.label][1] for obj in detected_objects])
    # print(res)
    tts = Speaker(voice="Zira")
    tts.speak(res)

if __name__ == '__main__':
    main()
