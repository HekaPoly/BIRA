from bira_components.camera import Camera
from bira_components.computer_vision import ComputerVision
from bira_components.micro import Micro
from bira_components.SLM_Manager import SLM_Manager
from bira_components.speech_to_text import SpeechToText
from bira_components.text_to_speech import TextToSpeech
from bira_components.uart_transmitter import UARTTransmitter
from time import sleep


class BIRA_Controller:
    def __init__(self):
        # TODO: Decouple components from mediator. We want to get rid of the mediator and have the controller directly manage the components.
        self.camera = Camera()
        self.computer_vision = ComputerVision(camera=self.camera) # TODO: We should not pass the camera to the computer vision. We should have a better way to share data between components. Maybe we can have a shared memory or a shared database where the camera can write the frames and the computer vision can read them.
        self.uart_transmitter = UARTTransmitter()
        self.micro = Micro()
        self.text_to_speech = TextToSpeech()
        self.speech_to_text = SpeechToText()
        # self.slm_manager = SLM_Manager()
    

    # Example methods to control the components. TODO: Implement actual logic for these methods.
    def sleep_mode(self):
        self.micro.wait_for_volume()
    
    def listen(self):
        self.micro.record()
        transcription = self.speech_to_text.transcribe()
        return transcription
    
    def vision(self):
        frame = self.camera.get_frame() # TODO: We need 3 frames.
        sl_object, dectection_labels = self.computer_vision.detect_objects(frame)
        return sl_object, dectection_labels
    
    def send_mechanical_command(self, command):
         # EXAMPLE: Simulate an execution process that takes some time and is successful
        sleep(2)
    
    def speak(self, text):
        self.text_to_speech.speak(text)
    
    def prompt_slm(self, data):
        # prompt = self.slm_manager.create_prompt(data) # TODO: Define what data we want to send to the SLM and how to create the prompt.
        # response = self.slm_manager.generate_response(prompt)
        response = "This is a response from the SLM." # Placeholder response
        return response
    
    def destroy(self):
        components = [
            self.camera,
            self.computer_vision,
            self.uart_transmitter,
            self.micro,
            self.text_to_speech,
            self.speech_to_text,
            # self.slm_manager
        ]

        for component in components :
            try:
                if component and hasattr(component,'destroy'):
                    component.destroy()
            except Exception as e:
                print(f"Failed to destroy {component} : {e}")


