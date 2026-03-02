from __future__ import annotations
from abc import ABC
import threading
import time


import argparse

from mediator import Mediator   
from camera import Camera
from computer_vision import ComputerVision
from uart_transmitter import UARTTransmitter
from micro import Micro
from text_to_speech import TextToSpeech
from speech_to_text import SpeechToText
    
class BIRAManager(Mediator):
    def __init__(self, opt):
        self.camera = Camera()
        self.computer_vision = ComputerVision(opt)
        self.uart_transmitter = UARTTransmitter()
        self.micro = Micro()
        self.text_to_speech = TextToSpeech()
        self.speech_to_text = SpeechToText()
        self.slm = None

    def speech_task(self):
        print("Starting speech recognition...")
        self.micro.record(duration=5)
        self.micro.save_recording('recording.wav')
        transcribe = self.speech_to_text.transcribe('recording.wav')
        print("Transcription:", transcribe)

    def vision_task(self,results):
            for _ in range(3):
                frame = self.camera.get_frame()
                detections = self.computer_vision.detect(frame)
                results.append(detections)
                time.sleep(0.05)
    
    def notify(self, sender: object, event: str) -> None:
        if event == "wait_for_volume":
            print("Waiting for volume..")
            self.micro.wait_for_volume()
            
        elif event == "volume_detected":
            if (self.camera.grab()):

                detections = []

                t_speech = threading.Thread(target=self.speech_task)
                t_vision = threading.Thread(target=self.vision_task, args=(detections,))

                t_speech.start()
                t_vision.start()

                t_speech.join()
                t_vision.join()
                
                print("Final detections:", detections)
                # Send command to SLM.
            
        if event == "wake_up":
            self.camera.open()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()
    
    manager = BIRAManager(opt)
    print("BIRA Manager initialized.")
    manager.notify(None, "wake_up")
    manager.notify(None, "volume_detected")
    # sleep(2000)  # Allow camera to warm up
    