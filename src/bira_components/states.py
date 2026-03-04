from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from time import sleep

from .codes import ListeningCode, VisionCode, PlanificationCode, ExecutionCode

if TYPE_CHECKING:
    from .manager import BIRA_Manager

class State(ABC):
    def __init__(self, bira_manager: BIRA_Manager):
        self.bira_manager = bira_manager
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def handle(self):
        pass

    @abstractmethod
    def decide_next_state(self):
        pass

class IdleState(State):
    def __str__(self):
        return "IdleState"

    def handle(self):
        # TODO: Implement actual sleep logic using self.bira_manager.micro
        # self.bira_manager.micro.sleep_mode()
        pass
    
    def decide_next_state(self):
        print("Wake up")
        self.bira_manager.change_state(ListeningState(self.bira_manager))
    
class ListeningState(State):
    def __str__(self):
        return "ListeningState"
    
    def handle(self):
        # TODO: Implement actual listening logic using self.bira_manager.micro and self.bira_manager.speech_to_text
        # self.bira_manager.micro.record()
        # result = self.bira_manager.speech_to_text.transcribe()
        # self.bira_manager.set_user_input(result)
        pass

    def decide_next_state(self):
        listening_code = self.bira_manager.get_data()["listening_code"]

        match listening_code:
            case ListeningCode.ERROR:
                print("Error occurred during listening processing.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case ListeningCode.NO_RESPONSE:
                print("Listening has not been done yet. System is not supposed to be in this state.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case ListeningCode.SUCCESS:
                print("Listening processing successful.")
                self.bira_manager.set_feedback("Vous m'avez demandé: {0}.").format(self.bira_manager.get_data()["user_input"])
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(VisionState(self.bira_manager))
            case ListeningCode.NO_INPUT:
                print("No voice input received.")
                self.bira_manager.set_feedback("Je n'ai pas entendu votre commande. Je vais me remettre en veille.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(IdleState(self.bira_manager))
    

class VisionState(State):
    def __str__(self):
        return "VisionState"
    
    def handle(self):
        # TODO: Perform actions specific to Vision state
        # sl_object, dectection_labels = self.bira_manager.computer_vision.detect_objects()
        # self.bira_manager.set_objects_detected(sl_object)
        # self.bira_manager.set_detection_labels(dectection_labels)
        pass
    
    def decide_next_state(self):
        vision_code = self.bira_manager.get_data()["vision_code"]
        match vision_code:
            case VisionCode.ERROR:
                print("Error occurred during vision processing.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case VisionCode.NO_RESPONSE:
                print("Vision has not been done yet. System is not supposed to be in this state.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case VisionCode.SUCCESS:
                print("Vision processing successful.")
                self.bira_manager.set_feedback("J'ai détecté les objets suivants: {0}.").format(self.bira_manager.get_data()["detection_labels"])
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(PlanningState(self.bira_manager))
            case VisionCode.NO_OBJECT_DETECTED:
                print("No objects detected.")
                self.bira_manager.set_feedback("Je n'ai détecté aucun objet. Veuillez réessayer.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ListeningState(self.bira_manager))

class PlanningState(State):
    def __str__(self):
        return "PlanningState"
    
    def handle(self):
        # TODO: Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions
        # If successful, set feedback and plan next actions
        pass

    def decide_next_state(self):
        planification_code = self.bira_manager.get_data()["planification_code"]

        match planification_code:
            case PlanificationCode.ERROR:
                print("Error occurred during planification processing.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case PlanificationCode.NO_RESPONSE:
                print("Planification has not been done yet. System is not supposed to be in this state.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case PlanificationCode.SUCCESS:
                print("Planification processing successful.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ExecutingState(self.bira_manager))
            case PlanificationCode.UNCLEAR_COMMAND:
                print("Unclear command. User needs to reformulate.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ListeningState(self.bira_manager))
            case PlanificationCode.INAPPROPRIATE_REQUEST:
                print("Inappropriate request. User needs to ask for a valid object.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ListeningState(self.bira_manager))
            case PlanificationCode.UNDETECTED_OBJECT:
                print("Object not detected. Vision needs to be redone.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.increment_counter()  # to avoid infinite loop in case of persistent undetected object
                self.bira_manager.change_state(VisionState(self.bira_manager))
            case PlanificationCode.IDLE:
                print("Idle state reached in planification. Transitioning to RespondingState with feedback.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(IdleState(self.bira_manager))

class ExecutingState(State):
    def __str__(self):
        return "ExecutingState"
    
    def handle(self):
        # Perform actions specific to Executing state
        # For example, send commands to actuators or perform a task based on the response generated in RespondingState
        sleep(5)
    
    def decide_next_state(self):
        execution_code = self.bira_manager.get_data()["execution_code"]

        match execution_code:
            case ExecutionCode.ERROR:
                print("Error occurred during execution processing.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case ExecutionCode.NO_RESPONSE:
                print("Execution has not been done yet. System is not supposed to be in this state.")
                self.bira_manager.change_state(ExitState(self.bira_manager))
            case ExecutionCode.SUCCESS:
                print("Execution processing successful.")
                self.bira_manager.set_feedback("J'ai exécuté la tâche demandée. Voulez-vous que je fasse autre chose ?")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ListeningState(self.bira_manager))
            case ExecutionCode.UNABLE_TO_MOVE:
                print("Unable to move. I might be blocked or there might be an obstacle.")
                self.bira_manager.set_feedback("Je n'ai pas pu atteindre l'objet. Il semble qu'il y ait un obstacle ou que je suis bloqué.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(IdleState(self.bira_manager))
            case ExecutionCode.UNREACHABLE_OBJECT:
                print("Unreachable object. The object might be out of my reach or might have been moved since the vision stage.")
                self.bira_manager.set_feedback("Je n'ai pas pu atteindre l'objet. Il semble que l'objet soit hors de ma portée ou qu'il ait été déplacé depuis la vision.")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.increment_counter()  # to avoid infinite loop in case of persistent undetected object
                self.bira_manager.change_state(VisionState(self.bira_manager))
            case ExecutionCode.OBJECT_DROPPED:
                print("Object dropped. I might have dropped the object during the execution.")
                self.bira_manager.set_feedback("J'ai laissé tomber l'objet pendant l'exécution. Je suis désolé. Voulez-vous que je réessaye ?")
                # TODO: Send feedback to user via text-to-speech
                self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])
                self.bira_manager.change_state(ListeningState(self.bira_manager))

class ExitState(State):
    def __str__(self):
        return "ExitState"
    
    def handle(self):
        # Perform actions specific to Exit state
        # For example, clean up resources, save state, or perform any necessary shutdown procedures
        print("Exiting the system. Cleaning up resources and shutting down.")
        exit(0)
    
    def decide_next_state(self):
        # No next state to transition to since this is the exit state
        pass