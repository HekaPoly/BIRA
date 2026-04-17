from types import SimpleNamespace
from time import sleep

from bira_components import history as bira_history
from bira_components.SLM_Manager import SLM_Manager
from bira_orchestration.enums import PlanificationCode


class _MockComputerVision:
    def __init__(self):
        # Common COCO labels used in default mock scene.
        self._labels = {
            39: "bottle",
            41: "cup",
            67: "cell phone",
        }

    def warmup(self):
        print("[MOCK:VISION] Warmup skipped.")

    def get_label_name(self, label_id: int) -> str:
        return self._labels.get(int(label_id), f"Unknown_{label_id}")

    @property
    def label_dict(self):
        return self._labels


class _MockDetectedObject:
    def __init__(self, obj_id: int, raw_label: int, confidence: int, position: list[float], bbox_2d: list[list[int]]):
        self.id = obj_id
        self.raw_label = raw_label
        self.confidence = confidence
        self.position = position
        self.bounding_box_2d = bbox_2d


class MockedBiraController:
    def __init__(self):
        print("[MOCK MODE] Hardware/audio components are mocked. SLM remains active.")
        self.camera = None
        self.computer_vision = _MockComputerVision()
        self.uart_transmitter = None
        self.micro = None
        self.text_to_speech = None
        self.speech_to_text = None
        self.slm_manager = SLM_Manager(mode="local")

    def preload_components(self):
        preload_steps = [
            ("language model", self.slm_manager.preload),
        ]

        preload_summary = []
        for label, preload in preload_steps:
            print(f"Preloading {label}...")
            try:
                preload()
                preload_summary.append({"component": label, "status": "ready"})
                bira_history.log_event(
                    "component_preload_result",
                    component="mock_controller",
                    target=label,
                    status="ready",
                )
            except Exception as exc:
                print(f"Unable to preload {label}: {exc}")
                preload_summary.append({"component": label, "status": "failed", "error": str(exc)})
                bira_history.log_event(
                    "component_preload_result",
                    component="mock_controller",
                    target=label,
                    status="failed",
                    error=str(exc),
                )

        bira_history.log_event(
            "component_preload_summary",
            component="mock_controller",
            ready_count=sum(1 for item in preload_summary if item.get("status") == "ready"),
            total_count=len(preload_summary),
            results=preload_summary,
        )

    def sleep_mode(self):
        input("[MOCK:WAKE] Press Enter to wake BIRA... ")

    def listen(self):
        transcription = input("[MOCK:USER_INPUT] ").strip()
        if transcription.lower() in {"q", "quit", "exit"}:
            return "stop"
        bira_history.log_conversation("user", transcription, source="mock_input")
        return transcription

    def vision(self):
        mock_objects = [
            _MockDetectedObject(
                obj_id=1,
                raw_label=39,
                confidence=91,
                position=[0.35, -0.12, 1.05],
                bbox_2d=[[100, 90], [160, 90], [100, 190], [160, 190]],
            ),
            _MockDetectedObject(
                obj_id=2,
                raw_label=39,
                confidence=88,
                position=[0.12, 0.05, 0.82],
                bbox_2d=[[260, 100], [320, 100], [260, 200], [320, 200]],
            ),
            _MockDetectedObject(
                obj_id=3,
                raw_label=41,
                confidence=86,
                position=[-0.10, 0.03, 0.72],
                bbox_2d=[[360, 120], [420, 120], [360, 200], [420, 200]],
            ),
        ]
        detection_labels = [obj.raw_label for obj in mock_objects]
        return SimpleNamespace(object_list=mock_objects), detection_labels

    def send_mechanical_command(self, command):
        label_name = "unknown"
        raw = getattr(command, "raw_label", None)
        if raw is not None and self.computer_vision is not None:
            label_name = self.computer_vision.get_label_name(int(raw))
        print(f"[MOCK:EXECUTION] Pretending to execute grasp action on: {label_name}")
        sleep(0.3)

    def speak(self, text):
        bira_history.log_conversation("assistant", text, source="mock_tts")
        print(f"\n=== BIRA_FEEDBACK ===\n{text}\n=====================\n")

    def prompt_slm(self, context):
        if context.user_inputs:
            # Keep only the latest user utterance here; dialog continuity is handled by SLM history.
            user_input = context.user_inputs[-1]
        else:
            user_input = None
        detected_objects = context.objects_detected if context.objects_detected else None

        bira_history.log_event(
            "planning_request_started",
            component="mock_controller",
            user_input=user_input or "",
            object_count=len(detected_objects or []),
            detection_labels=context.detection_labels or [],
        )

        self.slm_manager.set_transcription(user_input or "")
        self.slm_manager.set_detections(
            detection_labels=context.detection_labels,
            detected_objects=detected_objects,
            computer_vision=self.computer_vision,
        )

        response = self.slm_manager.run_inference()
        bira_history.log_event(
            "planning_request_completed",
            component="mock_controller",
            backend=self.slm_manager.last_backend,
            mode=response.get("mode"),
            request_scope=response.get("request_scope"),
            selected_candidate_index=response.get("selected_candidate_index"),
            selected_label=response.get("selected_label"),
            selected_label_id=response.get("selected_label_id"),
        )
        bira_history.log_conversation(
            "assistant_planning",
            response.get("response", ""),
            source="slm_manager",
            backend=self.slm_manager.last_backend,
            mode=response.get("mode"),
            request_scope=response.get("request_scope"),
            selected_candidate_index=response.get("selected_candidate_index"),
            selected_label=response.get("selected_label"),
            selected_label_id=response.get("selected_label_id"),
        )
        return response

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

        for component in components:
            try:
                if component and hasattr(component, "destroy"):
                    component.destroy()
            except Exception as exc:
                print(f"Failed to destroy {component}: {exc}")
