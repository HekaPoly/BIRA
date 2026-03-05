from __future__ import annotations
from dataclasses import dataclass, field

from enums import ListeningCode, VisionCode, PlanificationCode, ExecutionCode

# Object context is managed by the BIRA_CONTROLLER, response codes are managed by states.
@dataclass
class BIRA_Context:
    objects_detected: list = field(default_factory=list)
    detection_labels: list = field(default_factory=list)
    user_inputs: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)
    object_selected: object | None = None

    listening_code: ListeningCode = ListeningCode.NO_RESPONSE
    vision_code: VisionCode = VisionCode.NO_RESPONSE
    planification_code: PlanificationCode = PlanificationCode.NO_RESPONSE
    execution_code: ExecutionCode = ExecutionCode.NO_RESPONSE