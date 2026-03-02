import argparse

from bira_componants.mediator import BiraMediator   
from bira_componants.camera import Camera
from bira_componants.computer_vision import ComputerVision
from bira_componants.uart_transmitter import UARTTransmitter
from bira_componants.micro import Micro
from bira_componants.text_to_speech import TextToSpeech
from bira_componants.speech_to_text import SpeechToText
from bira_componants.SLM_Manager import SLM_Manager
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n_resna.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()
    
    
    mediator = BiraMediator()
    camera = Camera(mediator=mediator)
    computer_vision = ComputerVision(opt, mediator=mediator)
    uart_transmitter = UARTTransmitter(mediator=mediator)
    micro = Micro(mediator=mediator)
    text_to_speech = TextToSpeech(mediator=mediator)
    speech_to_text = SpeechToText(mediator=mediator, language="fr")
    slm = SLM_Manager(mediator=mediator)
    
    mediator.send(mediator, {"initialize_components": None})
    mediator.send(mediator, {"sleep_mode": None})
    
    print(mediator.handlers)
    # mediator.send_to(target=speech_to_text, sender=mediator, message={"transcribe_1": "recording.wav"})     
    mediator.run()
    
    
if __name__ == "__main__":
    main()
    