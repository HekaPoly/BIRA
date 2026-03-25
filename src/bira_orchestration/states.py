from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from bira_orchestration.enums import (
    ListeningCode,
    VisionCode,
    PlanificationCode,
    ExecutionCode,
    StateCode,
)

if TYPE_CHECKING:
    from manager import BIRA_Manager

class State(ABC):
    code: StateCode
    
    def __init__(self, bira_manager: BIRA_Manager):
        self.bira_manager = bira_manager
    
    def handle(self):
        self._prepare()
        self._handle()
        self._decide_next_state()
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def _prepare(self):
        pass

    @abstractmethod
    def _handle(self):
        pass

    @abstractmethod
    def _decide_next_state(self):
        pass

class IdleState(State):
    code = StateCode.IDLE
    
    def __str__(self):
        return "IdleState"

    def _prepare(self):
        self.bira_manager.reset_data()

    def _handle(self):
        # TODO: Implement actual sleep logic
        self.bira_manager.controller.sleep_mode()
    
    def _decide_next_state(self):
        print("Wake up")
        feedback = "Je suis réveillé. Que puis-je faire pour vous ?"
        self.bira_manager.add_feedback(feedback)
        self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(StateCode.LISTENING)
    
class ListeningState(State):
    code = StateCode.LISTENING
    
    def __str__(self):
        return "ListeningState"

    def _prepare(self):
        self.bira_manager.prepare_listening()
        
    def _handle(self):
        # TODO: Implement actual listening logic
        transcription = self.bira_manager.controller.listen()
        self.bira_manager.add_user_input(transcription)
        self.bira_manager.get_data().listening_code = ListeningCode.SUCCESS

    def _decide_next_state(self):
        listening_code = self.bira_manager.get_data().listening_code
        feedback = None
        new_state = StateCode.EXIT

        match listening_code:
            case ListeningCode.ERROR:
                print("Error occurred during listening processing.")
                new_state = StateCode.EXIT
            case ListeningCode.NO_RESPONSE:
                print("Listening has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ListeningCode.SUCCESS:
                print("Listening processing successful.")
                feedback = f"Vous m'avez demandé: {self.bira_manager.get_last_user_input()}."
                new_state = StateCode.VISION
            case ListeningCode.NO_INPUT:
                print("No voice input received.")
                feedback = "Je n'ai pas entendu votre commande. Je vais me remettre en veille."
                new_state = StateCode.IDLE
            case _:
                print("Unknown listening code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT
        
        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)

class VisionState(State):
    code = StateCode.VISION
    
    def __str__(self):
        return "VisionState"
    
    def _prepare(self):
        self.bira_manager.prepare_vision()

    def _handle(self):
        # TODO: Implement actual vision logic
        sl_object, detection_labels = self.bira_manager.controller.vision()
        context = self.bira_manager.get_data()
        context.objects_detected = sl_object.object_list
        context.detection_labels = detection_labels
        context.vision_code = VisionCode.SUCCESS
    
    def _decide_next_state(self):
        vision_code = self.bira_manager.get_data().vision_code
        feedback = None
        new_state = StateCode.EXIT

        match vision_code:
            case VisionCode.ERROR:
                print("Error occurred during vision processing.")
                new_state = StateCode.EXIT
            case VisionCode.NO_RESPONSE:
                print("Vision has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case VisionCode.SUCCESS:
                print("Vision processing successful.")
                feedback = f"J'ai détecté les objets suivants: {', '.join(str(self.bira_manager.get_data().detection_labels))}."
                new_state = StateCode.PLANNING
            case VisionCode.NO_OBJECT_DETECTED:
                print("No objects detected.")
                feedback = "Je n'ai détecté aucun objet. Veuillez réessayer."
                new_state = StateCode.LISTENING
            case _:
                print("Unknown vision code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT
        
        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)
    
class PlanningState(State):
    code = StateCode.PLANNING
    
    def __str__(self):
        return "PlanningState"
    
    def _prepare(self):
        self.bira_manager.prepare_planning()

    def _handle(self):
        # TODO: Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions
        # If successful, set feedback and plan next actions
        context = self.bira_manager.get_data()
        response = self.bira_manager.controller.prompt_slm(context)
        feedback = response.get("response", "Je n'ai pas compris la demande.")
        mode = response.get("mode", "clarification")

        self.bira_manager.add_feedback(feedback)
        
        if mode == "confirmation":
            context.object_selected = context.objects_detected[0] if context.objects_detected else None
            print(context.object_selected)
            context.planification_code = PlanificationCode.SUCCESS
        elif mode == "stop":
            context.planification_code = PlanificationCode.IDLE
        else:
            context.planification_code = PlanificationCode.UNCLEAR_COMMAND

    def _decide_next_state(self):
        planification_code = self.bira_manager.get_data().planification_code
        feedback = None
        new_state = StateCode.EXIT

        match planification_code:
            case PlanificationCode.ERROR:
                print("Error occurred during planification processing.")
                new_state = StateCode.EXIT
            case PlanificationCode.NO_RESPONSE:
                print("Planification has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case PlanificationCode.SUCCESS:
                print("Planification processing successful.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.EXECUTING
            case PlanificationCode.UNCLEAR_COMMAND:
                print("Unclear command. User needs to reformulate.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.INAPPROPRIATE_REQUEST:
                print("Inappropriate request. User needs to ask for a valid object.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.LISTENING
            case PlanificationCode.UNDETECTED_OBJECT:
                print("Object not detected. Vision needs to be redone.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.VISION
            case PlanificationCode.IDLE:
                print("Idle state reached in planification. Transitioning to RespondingState with feedback.")
                feedback = self.bira_manager.get_last_feedback()
                new_state = StateCode.IDLE
            case _:
                print("Unknown planification code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT
        
        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)

class ExecutingState(State):
    code = StateCode.EXECUTING
    
    def __str__(self):
        return "ExecutingState"
    
    def _prepare(self):
        self.bira_manager.prepare_execution()
    
    def _handle(self):
        # Perform actions specific to Executing state
        # For example, send commands to actuators or perform a task based on the response generated in RespondingState
        
        self.bira_manager.controller.send_mechanical_command(self.bira_manager.get_data().object_selected)
        self.bira_manager.get_data().execution_code = ExecutionCode.SUCCESS
    
    def _decide_next_state(self):
        execution_code = self.bira_manager.get_data().execution_code
        feedback = None
        new_state = StateCode.EXIT

        match execution_code:
            case ExecutionCode.ERROR:
                print("Error occurred during execution processing.")
                new_state = StateCode.EXIT
            case ExecutionCode.NO_RESPONSE:
                print("Execution has not been done yet. System is not supposed to be in this state.")
                new_state = StateCode.EXIT
            case ExecutionCode.SUCCESS:
                print("Execution processing successful.")
                feedback = "J'ai exécuté la tâche demandée. Voulez-vous que je fasse autre chose ?"
                new_state = StateCode.LISTENING
            case ExecutionCode.UNABLE_TO_MOVE:
                print("Unable to move. I might be blocked or there might be an obstacle.")
                feedback = "Je n'ai pas pu atteindre l'objet. Il semble qu'il y ait un obstacle ou que je suis bloqué."
                new_state = StateCode.IDLE
            case ExecutionCode.UNREACHABLE_OBJECT:
                print("Unreachable object. The object might be out of my reach or might have been moved since the vision stage.")
                feedback = "Je n'ai pas pu atteindre l'objet. Il semble que l'objet soit hors de ma portée ou qu'il ait été déplacé depuis la vision."
                new_state = StateCode.VISION
            case ExecutionCode.OBJECT_DROPPED:
                print("Object dropped. I might have dropped the object during the execution.")
                feedback = "J'ai laissé tomber l'objet pendant l'exécution. Je suis désolé. Voulez-vous que je réessaye ?"
                new_state = StateCode.LISTENING
            case _:
                print("Unknown execution code. Transitioning to exit state for safety.")
                new_state = StateCode.EXIT
                
        if feedback:
            self.bira_manager.add_feedback(feedback)
            self.bira_manager.controller.speak(feedback)
        self.bira_manager.change_state(new_state)

class ExitState(State):
    code = StateCode.EXIT
    
    def __str__(self):
        return "ExitState"

    def _prepare(self):
        # Keep the information about the error that caused the exit in the feedback, to be able to send it to the user or save it for later analysis
        pass
    
    def _handle(self):
        # Perform actions specific to Exit state
        # For example, clean up resources, save state, or perform any necessary shutdown procedures
        feedback = "Une erreur est survenue. Je vais devoir m'arrêter. Veuillez vérifier le système et réessayer."
        self.bira_manager.add_feedback(feedback)
        self.bira_manager.controller.speak(feedback)
        self.bira_manager.controller.destroy()
    
    def _decide_next_state(self):
        # No next state to transition to since this is the exit state
        print("Exiting the system. Cleaning up resources and shutting down.")
        exit(0)