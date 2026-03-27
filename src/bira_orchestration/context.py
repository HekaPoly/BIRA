from __future__ import annotations
from dataclasses import dataclass, field

from bira_orchestration.enums import ListeningCode, VisionCode, PlanificationCode, ExecutionCode

# Object context is managed by the BIRA_CONTROLLER, response codes are managed by states.
@dataclass
class BiraContext:
    objects_detected: list = field(default_factory=list)
    detection_labels: list = field(default_factory=list)
    user_inputs: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)
    object_selected: object | None = None

    listening_code: ListeningCode = ListeningCode.NO_RESPONSE
    vision_code: VisionCode = VisionCode.NO_RESPONSE
    planification_code: PlanificationCode = PlanificationCode.NO_RESPONSE
    execution_code: ExecutionCode = ExecutionCode.NO_RESPONSE

    def reset_all(self) -> None:
        self.objects_detected.clear()
        self.detection_labels.clear()
        self.user_inputs.clear()
        self.feedbacks.clear()
        self.object_selected = None
        self.reset_codes()

    def reset_codes(self) -> None:
        self.listening_code = ListeningCode.NO_RESPONSE
        self.vision_code = VisionCode.NO_RESPONSE
        self.planification_code = PlanificationCode.NO_RESPONSE
        self.execution_code = ExecutionCode.NO_RESPONSE

    def reset_for_listening(self) -> None:
        self.vision_code = VisionCode.NO_RESPONSE

    def reset_for_vision(self) -> None:
        self.objects_detected.clear()
        self.detection_labels.clear()
        self.vision_code = VisionCode.NO_RESPONSE

    def reset_for_planning(self) -> None:
        self.planification_code = PlanificationCode.NO_RESPONSE

    def reset_for_execution(self) -> None:
        self.execution_code = ExecutionCode.NO_RESPONSE