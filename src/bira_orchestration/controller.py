from bira_components.camera import Camera
from bira_components.computer_vision import ComputerVision
from bira_components.micro import Micro
from bira_components.SLM_Manager import SLM_Manager
from bira_components.speech_to_text import SpeechToText
from bira_components.text_to_speech import TextToSpeech
from bira_components.uart_transmitter import UARTTransmitter
from time import sleep


class BiraController:
    def __init__(self):
        self.camera = Camera()
        # TODO: We should not pass the camera to the computer vision. We should have a better way to share data between
        #  components. Maybe we can have a shared memory or a shared database where the camera can write the frames and
        #  the computer vision can read them.
        self.computer_vision = ComputerVision(camera=self.camera)
        # self.uart_transmitter = UARTTransmitter()
        self.micro = Micro()
        self.text_to_speech = TextToSpeech()
        self.speech_to_text = SpeechToText()
        self.slm_manager = SLM_Manager(mode="local")

    def preload_components(self):
        preload_steps = [
            ("text-to-speech engine", self.text_to_speech.preload),
            ("speech-to-text model", self.speech_to_text.preload),
            ("vision model", self.computer_vision.warmup),
            ("language model", self.slm_manager.preload),
        ]

        for label, preload in preload_steps:
            print(f"Preloading {label}...")
            try:
                preload()
            except Exception as exc:
                print(f"Unable to preload {label}: {exc}")

    def sleep_mode(self):
        self.micro.wait_for_volume()
    
    def listen(self):
        self.micro.record()
        transcription = self.speech_to_text.transcribe()
        self.micro.clear_recording()
        return transcription
    
    def vision(self):
        self.camera.open()
        
        with self.camera:
            if self.camera.grab():
                frame = self.camera.get_frame()
                if frame is not None:
                    sl_object, detection_labels = self.computer_vision.detect_objects(frame)
                else:
                    print("No frame")
                    self.camera.close()
                    return None
            else:
                print("No camera")
                self.camera.close()
                return None
        
        self.camera.close()
        return sl_object, detection_labels
    
    def send_mechanical_command(self, command):
         # EXAMPLE: Simulate an execution process that takes some time and is successful
        sleep(2)
    
    def speak(self, text):
        print(f"Bira Speaking: {text}")
        self.text_to_speech.speak(text)
    
    def prompt_slm(self, context):
        user_input = context.user_inputs[-1] if context.user_inputs else None
        detected_objects = getattr(context.objects_detected, "object_list", None)

        self.slm_manager.set_transcription(user_input or "")
        self.slm_manager.set_detections(
            detection_labels=context.detection_labels,
            detected_objects=detected_objects,
        )

        return self.slm_manager.run_inference()
    
    def destroy(self):
        components = [
            self.camera,
            self.computer_vision,
            # self.uart_transmitter,
            self.micro,
            self.text_to_speech,
            self.speech_to_text,
            self.slm_manager,
        ]

        for component in components :
            try:
                if component and hasattr(component,'destroy'):
                    component.destroy()
            except Exception as e:
                print(f"Failed to destroy {component} : {e}")


