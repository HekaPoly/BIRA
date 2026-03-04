from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from time import sleep

if TYPE_CHECKING:
    from .manager import BIRA_Manager
# from .manager import BIRA_Manager

class State(ABC):
    def __init__(self, bira_manager: BIRA_Manager):
        self.bira_manager = bira_manager
    
    @abstractmethod
    def handle(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

class IdleState(State):
    def handle(self):
        self.bira_manager.set_user_wants_idle_state(False)
        print("Entering Idle State")
        
        # self.bira_manager.micro.sleep_mode()
        print("Wake up")
        self.bira_manager.change_state(ListeningState(self.bira_manager))
    
    def __str__(self):
        return "IdleState"
    
class ListeningState(State):
    def handle(self):
        print("Entering Listening State")
        self.bira_manager.counter += 1
        print(f"Counter incremented to: {self.bira_manager.counter}")
        if self.bira_manager.counter > 3:
            print("Already been in ListeningState three times. Transitioning to ExitState instead.")
            self.bira_manager.change_state(ExitState(self.bira_manager))
            return


        # self.bira_manager.micro.record()
        # result = self.bira_manager.speech_to_text.transcribe()
        # self.bira_manager.set_user_input(result)
        # self.bira_manager.change_state(VisionState(self.bira_manager))

    def __str__(self):
        return "ListeningState"

class VisionState(State):
    def handle(self):
        print("Entering Vision State")
        # Perform actions specific to Vision state
        # sl_object, dectection_labels = self.bira_manager.computer_vision.detect_objects()
        # self.bira_manager.set_objects_detected(sl_object)
        # self.bira_manager.set_detection_labels(dectection_labels)
        # self.bira_manager.change_state(PlanningState(self.bira_manager))
    
    def __str__(self):
        return "VisionState"

class PlanningState(State):
    def handle(self):
        print("Entering Planning State")
        # Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions

    def __str__(self):
        return "PlanningState"
class RespondingState(State):
    def handle(self):
        print("Entering Responding State")
        # Perform actions specific to Responding state
        # For example, generate a response based on the analysis from PlanningState
        # self.bira_manager.text_to_speech.speak_sync(self.bira_manager.get_data()["feedback"])

        # if self.bira_manager.get_data()["response_code"] == 0:
        #     self.bira_manager.change_state(ExecutingState(self.bira_manager))
        # elif self.bira_manager.get_data()["response_code"] == 1:
        #     self.bira_manager.change_state(ErrorState(self.bira_manager))
        # else:
        #     self.bira_manager.change_state(IdleState(self.bira_manager))
    
    def __str__(self):
        return "RespondingState"

class ExecutingState(State):
    def handle(self):
        print("Entering Executing State")
        # Perform actions specific to Executing state
        # For example, send commands to actuators or perform a task based on the response generated in RespondingState
        sleep(5)
        # self.bira_manager.change_state(SuccessState(self.bira_manager))

    def __str__(self):
        return "ExecutingState"
class SuccessState(State):
    def handle(self):
        print("Entering Success State")
        # Perform actions specific to Success state
        # For example, log the successful completion of a task or notify the user
        # self.bira_manager.text_to_speech.speak_sync("Task completed successfully.")

    def __str__(self):
        return "SuccessState"
class ErrorState(State):
    def handle(self):
        print("Entering Error State")
        #TODO: Perform actions specific to Error state

    def __str__(self):
        return "ErrorState"
    
class ExitState(State):
    def handle(self):
        print("Entering Exit State")
        # Perform actions specific to Exit state
        # For example, clean up resources, save state, or perform any necessary shutdown procedures

    def __str__(self):
        return "ExitState"