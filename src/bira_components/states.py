from abc import ABC, abstractmethod

class State(ABC):
    def __init__(self, bira_manager):
        self.bira_manager = bira_manager
    
    @abstractmethod
    def handle(self, context):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

class IdleState(State):
    def handle(self, context):
        print("Entering Idle State")
        # Perform actions specific to Idle state
        # For example, wait for a command or event to transition to another state

    def get_name(self):
        return "IdleState"
    
class ListeningState(State):
    def handle(self, context):
        print("Entering Listening State")
        # Perform actions specific to Listening state
        # For example, start recording audio or listen for a specific command

    def get_name(self):
        return "ListeningState"

class VisionState(State):
    def handle(self, context):
        print("Entering Vision State")
        # Perform actions specific to Vision state
        # For example, start processing camera input or perform object detection

    def get_name(self):
        return "VisionState"
    
class PlanningState(State):
    def handle(self, context):
        print("Entering Planning State")
        # Perform actions specific to Planning state
        # For example, analyze data from VisionState and make decisions

    def get_name(self):
        return "PlanningState"

class RespondingState(State):
    def handle(self, context):
        print("Entering Responding State")
        # Perform actions specific to Responding state
        # For example, generate a response based on the analysis from PlanningState

    def get_name(self):
        return "RespondingState"
    
class ExecutingState(State):
    def handle(self, context):
        print("Entering Executing State")
        # Perform actions specific to Executing state
        # For example, send commands to actuators or perform a task based on the response generated in RespondingState

    def get_name(self):
        return "ExecutingState"

class SuccessState(State):
    def handle(self, context):
        print("Entering Success State")
        # Perform actions specific to Success state
        # For example, log the successful completion of a task or notify the user

    def get_name(self):
        return "SuccessState"

class ErrorState(State):
    def handle(self, context):
        print("Entering Error State")
        # Perform actions specific to Error state
        # For example, log the error, attempt recovery, or notify the user

    def get_name(self):
        return "ErrorState"