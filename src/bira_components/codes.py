from enum import IntEnum


class ListeningCode(IntEnum):
    """Codes for the listening stage.

    -1: error
     0: no response (yet)
     1: success
     2: no input
    """

    ERROR = -1
    NO_RESPONSE = 0
    SUCCESS = 1
    NO_INPUT = 2


class VisionCode(IntEnum):
    """Codes for the vision stage.

    -1: error
     0: no response (yet)
     1: success
     2: no object detected
    """

    ERROR = -1
    NO_RESPONSE = 0
    SUCCESS = 1
    NO_OBJECT_DETECTED = 2


class PlanificationCode(IntEnum):
    """Codes for the planification stage.

    -1: error
     0: no response (yet)
     1: success
     2: unclear command
     3: object not detected in vision
     4: inappropriate request (e.g. user asked for something that is not an object, or that cannot be manipulated)
     5: idle (user want to stop the process, or no input received)
    """

    ERROR = -1
    NO_RESPONSE = 0
    SUCCESS = 1
    UNCLEAR_COMMAND = 2
    INAPPROPRIATE_REQUEST = 3
    UNDETECTED_OBJECT = 4
    IDLE = 5


class ExecutionCode(IntEnum):
    """Codes for the execution stage.

    -1: error
     0: no response (yet)
     1: success
     2: unable to move (e.g. the robot is stuck or an object is blocking the way)
     3: unreachable location (e.g. the object is detected but the robot cannot reach it, or the object has been moved since the vision stage)
     4: object dropped (e.g. the robot tried to move an object but it was dropped during the movement)
    """

    ERROR = -1
    NO_RESPONSE = 0
    SUCCESS = 1
    UNABLE_TO_MOVE = 2
    UNREACHABLE_OBJECT = 3
    OBJECT_DROPPED = 4
