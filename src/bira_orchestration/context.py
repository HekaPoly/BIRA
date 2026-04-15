from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

ZedObjectData = Any # When not mocked, this will be the actual ZedObjectData class (pyzed.sl.ObjectData) from the ZED SDK. When mocked, it can be any structure that mimics the necessary attributes (e.g., position, dimensions, raw_label, bounding_box).

from bira_orchestration.enums import ListeningCode, VisionCode, PlanificationCode, ExecutionCode

# Object context is managed by the BIRA_CONTROLLER, response codes are managed by states.
@dataclass
class BiraContext:
    objects_detected: list[ZedObjectData] = field(default_factory=list)
    detection_labels: list[int] = field(default_factory=list)
    user_inputs: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)
    object_selected: ZedObjectData | None = None
    
    listening_code: ListeningCode = ListeningCode.NO_RESPONSE
    vision_code: VisionCode = VisionCode.NO_RESPONSE
    planification_code: PlanificationCode = PlanificationCode.NO_RESPONSE
    execution_code: ExecutionCode = ExecutionCode.NO_RESPONSE
    skip_vision_for_current_input: bool = False
    route_mode_hint: str | None = None
    feedback_source: str = "state_log"

    def reset_all(self) -> None:
        self.objects_detected.clear()
        self.detection_labels.clear()
        self.user_inputs.clear()
        self.feedbacks.clear()
        self.object_selected = None
        self.skip_vision_for_current_input = False
        self.route_mode_hint = None
        self.feedback_source = "state_log"
        self.reset_codes()

    def reset_codes(self) -> None:
        self.listening_code = ListeningCode.NO_RESPONSE
        self.vision_code = VisionCode.NO_RESPONSE
        self.planification_code = PlanificationCode.NO_RESPONSE
        self.execution_code = ExecutionCode.NO_RESPONSE

    def reset_for_listening(self) -> None:
        self.vision_code = VisionCode.NO_RESPONSE
        self.skip_vision_for_current_input = False
        self.route_mode_hint = None
        self.feedback_source = "state_log"

    def reset_for_vision(self) -> None:
        self.objects_detected.clear()
        self.detection_labels.clear()
        self.vision_code = VisionCode.NO_RESPONSE

    def reset_for_planning(self) -> None:
        self.planification_code = PlanificationCode.NO_RESPONSE

    def reset_for_execution(self) -> None:
        self.execution_code = ExecutionCode.NO_RESPONSE

    def clear_vision_context(self) -> None:
        self.objects_detected.clear()
        self.detection_labels.clear()
        self.object_selected = None
        self.skip_vision_for_current_input = False
        self.route_mode_hint = None