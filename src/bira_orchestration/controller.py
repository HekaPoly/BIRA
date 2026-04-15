from time import sleep

from bira_components.camera import Camera
from bira_components.computer_vision import ComputerVision
from bira_components.micro import Micro
from bira_components.SLM_Manager import SLM_Manager
from bira_components.speech_to_text import SpeechToText
from bira_components.text_to_speech import TextToSpeech
from bira_components.uart_transmitter import UARTTransmitter
from bira_orchestration.enums import PlanificationCode


class BiraController:
    def __init__(self):
        self.camera = Camera()
        # TODO: We should not pass the camera to the computer vision. We should have a better way to share data between
        #  components. Maybe we can have a shared memory or a shared database where the camera can write the frames and
        #  the computer vision can read them.
        self.computer_vision = ComputerVision(camera=self.camera)
        self.uart_transmitter = None
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
    
    def speak(self, text, source: str = "state_log"):
        source_tag = {
            "state_log": "State log",
            "fallback_feedback": "Fallback feedback",
            "slm_feedback": "SLM_feedback",
        }.get(source, source)
        print(f"[{source_tag}] Bira Speaking: {text}")
        self.text_to_speech.speak(text)
    
    def prompt_slm(self, context):
        if context.user_inputs:
            # Keep only the latest user utterance here; dialog continuity is handled by SLM history.
            user_input = context.user_inputs[-1]
        else:
            user_input = None
        mode_hint = context.route_mode_hint
        if context.skip_vision_for_current_input and mode_hint in {
            "conversing",
            "out_of_scope",
            "repeat",
            "stop",
            "inappropriate",
            "unclear_action",
        }:
            return self.slm_manager.respond_from_mode_hint(
                mode=mode_hint,
                transcription=user_input or "",
                detections=[],
            )

        detected_objects = context.objects_detected if context.objects_detected else None

        self.slm_manager.set_transcription(user_input or "")
        self.slm_manager.set_detections(
            detection_labels=context.detection_labels,
            detected_objects=detected_objects,
            computer_vision=self.computer_vision,
        )

        return self.slm_manager.run_inference()

    def route_request(self, context):
        user_input = context.user_inputs[-1] if context.user_inputs else ""
        return self.slm_manager.route_request(user_input)
    
    def destroy(self):
        components = [
            self.camera,
            self.computer_vision,
            self.uart_transmitter,
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


