import argparse

from bira_componants.mediator import BiraMediator   
from bira_componants.camera import Camera
from bira_componants.computer_vision import ComputerVision
from bira_componants.uart_transmitter import UARTTransmitter
from bira_componants.micro import Micro
from bira_componants.text_to_speech import TextToSpeech
from bira_componants.speech_to_text import SpeechToText
    
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolov8n.pt', help='model.pt path(s)')
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
    speech_to_text = SpeechToText(mediator=mediator)
    slm = None
    
    await mediator.send(mediator, {"initialize_components":None})
    await mediator.send(mediator, {"sleep_mode": None})
    asyncio.create_task(mediator.run())
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    