from abc import ABC, abstractmethod

from .bira_manager import BIRA_manager

class State(ABC):
    def __init__(self, bira_manager : BIRA_manager):
        self.bira_manager = bira_manager
    
    @abstractmethod
    def handle(self, context):
        pass

class IdleState(State):
    def handle(self, context):
        print("Entering Idle State")
        
        self.bira_manager.micro.sleep_mode()
        print("Wake up")
        self.bira_manager.change_state(ListeningState(self.bira_manager))
    
class ListeningState(State):
    def handle(self, context):
        print("Entering Listening State")
        self.bira_manager.micro.start_transcription()
        self.bira_manager.change_state(VisionState(self.bira_manager))

class VisionState(State):
    def handle(self, context):
        print("Entering Vision State")
        # Perform actions specific to Vision state
        # For example, start processing camera input or perform object detection

    
class PlanningState(State):
    def handle(self, context):
        print("Entering Planning State")
        # Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions

class RespondingState(State):
    def handle(self, context):
        print("Entering Responding State")
        # Perform actions specific to Responding state
        # For example, generate a response based on the analysis from PlanningState
    
class ExecutingState(State):
    def handle(self, context):
        print("Entering Executing State")
        # Perform actions specific to Executing state
        # For example, send commands to actuators or perform a task based on the response generated in RespondingState

class SuccessState(State):
    def handle(self, context):
        print("Entering Success State")
        # Perform actions specific to Success state
        # For example, log the successful completion of a task or notify the user

class ErrorState(State):
    def handle(self, context):
        print("Entering Error State")
        # Perform actions specific to Error state
        # For example, log the error, attempt recovery, or notify the user
