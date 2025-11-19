import multiprocessing
from scipy.stats import trim_mean
import speech_to_text
import numpy as np
from text_viewer import TextViewer
from time import sleep
from tts import Speaker
import utils
import history as history
import argparse
import torch
import math
import faulthandler
import uart_transmitter
from enum import Enum
from computer_vision import ComputerVision
from camera import Camera
import pyzed.sl as sl
import cv2

faulthandler.enable()

class Mode(Enum):
    FIX_THRESHOLD = 0
    TRIMMED = 1

def find_angle(coordinated_target_list:np.ndarray, mode: Mode = Mode.FIX_THRESHOLD) -> int:
    x = z = 0
    if mode == Mode.TRIMMED:
        x = trim_mean(coordinated_target_list[:,0], proportiontocut=0.1)
        z = trim_mean(coordinated_target_list[:,2], proportiontocut=0.1)
    elif mode == Mode.FIX_THRESHOLD:
        x = history.get_distance(coordinated_target_list[:,0])
        z = history.get_distance(coordinated_target_list[:,2])
    
    angle_rad = math.atan(x / z)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def run_test_motors():
    while True:
            try:
                a = input("Enter angle: ")
                b = input("Enter motor ID: ")
                if input("Do you want to enter velocity? (y/N): ").strip().lower() == 'y':
                    c = input("Enter velocity %: ")
                else:
                    c = 80
                uart_transmitter.send_data_through_UART(int(a), int(b), int(c))
                print("Data sent successfully.\n")
            except ValueError:
                print("Invalid input. Please enter numeric values.\n\n")           

def run_stt():
    def run_text_window(queue: multiprocessing.Queue):
        app = TextViewer(queue)
        app.open()
        
    text_queue = multiprocessing.Queue()
    text_process = multiprocessing.Process(target=run_text_window, args=(text_queue,))
    text_process.start()

    text_queue.put("Je suis BIRA, le bras robotique de HEKA.")
    sleep(3)
    
    while True:
        text_queue.put({"text": "Dis moi quelque chose!", "countdown": 10})
        text = speech_to_text.transcribe_for(10)
        text_queue.put({"text": f"Tu as dis: \n{text}", "countdown": 10})
        sleep(10)

def run_bira_sequence(opt): 
    text = speech_to_text.transcribe_directly()
    print(text)
    label = utils.string_to_label(text)
    print(label)   
    
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
    input("Press ENTER to begin recording")

    cv = ComputerVision(opt)
    stt_res = speech_to_text.transcribe_directly()

    cam = Camera()
    cam.open()
    detected_objects = []

    try:
        while True:
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                frame = cam.get_frame()
                if frame is not None:
                    objects = cv.detect(frame)
                    
                    for obj in objects.object_list:
                        if len(obj.bounding_box) > 0 and not np.isnan(obj.position).any():
                            detected_objects.append({
                                'label_id': int(obj.raw_label),
                                'confidence': float(obj.confidence),
                                'position': [float(obj.position[0]), float(obj.position[1]), float(obj.position[2])]
                            })

            key = cv2.waitKey(1)
            if key == 27:
                break
    finally:
        cam.close()
    
    print(f"\nDetected {len(detected_objects)} objects:")
    print(detected_objects)
    print(stt_res)

    # Feed the data to llama3.2 model

    example_res = "Voici le Text-to-Speech de BIRA."
    tts = Speaker()
    tts.speak(example_res)

if __name__ == '__main__':
    main()
