"""Simplified SLM_Manager using Ollama's native capabilities."""

from __future__ import annotations

import json
import os
from typing import Optional

from ollama import Client

from bira_components.slm_controller import SLM_Controller
from bira_components.slm_formatter import SLM_Formatter
from bira_components.tensorrt_manager import TensorRT_Manager


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SLM_Manager:
    """Coordinates routing, vision, and SLM inference."""

    def __init__(
        self,
        model_name: str = "bira-assistant",
        temperature: float = 0.3,
        mode: str = "local",
        api_key: Optional[str] = None,
        tensorrt_engine_dir: Optional[str] = None,
        debug: Optional[bool] = None,
        stream_json: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.mode = mode
        self.history: list[dict] = []
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.debug = _env_flag("SLM_DEBUG", default=False) if debug is None else debug
        self.stream_json = _env_flag("SLM_STREAM", default=True) if stream_json is None else stream_json
        self.num_predict = int(os.getenv("SLM_NUM_PREDICT", "900"))
        self.pending_label: Optional[str] = None
        self.prefer_tensorrt = True

        if mode == "cloud":
            if not api_key:
                raise ValueError("API key required for cloud mode")
            self.client = Client(
                host="https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            self.client = Client(host="http://localhost:11434")

        self.formatter = SLM_Formatter(response_schema=SLM_Formatter.RESPONSE_SCHEMA)
        self.chat_controller = SLM_Controller(
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            debug=self.debug,
        )
        self.tensorrt_manager = TensorRT_Manager(
            mode=self.mode,
            prefer_tensorrt=self.prefer_tensorrt,
            engine_dir=tensorrt_engine_dir,
        )
        self.tensorrt_manager.initialize()
        self.prefer_tensorrt = self.tensorrt_manager.prefer_tensorrt

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def reset_task_context(self) -> None:
        self.history = []
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.pending_label = None

    def set_transcription(self, transcription: str) -> None:
        self.transcription = transcription

    @staticmethod
    def _extract_position_xyz(position) -> Optional[list[float]]:
        """Normalize a position container to [x, y, z] or None."""
        if position is None:
            return None

        try:
            if len(position) < 3:
                return None
            return [float(position[0]), float(position[1]), float(position[2])]
        except Exception:
            return None

    def set_detections(
        self,
        detection_labels: Optional[list[int]] = None,
        detected_objects: Optional[list] = None,
        computer_vision=None,
    ) -> None:
        resolved: list[str] = []
        candidates: list[dict] = []

        if detected_objects is not None and computer_vision is not None:
            for index, obj in enumerate(detected_objects):
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                label_id = int(raw)
                label_name = computer_vision.get_label_name(label_id)
                resolved.append(label_name)

                position = getattr(obj, "position", None)
                pos_value = self._extract_position_xyz(position)

                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": label_id,
                        "position": pos_value,
                    }
                )
        elif detection_labels is not None and computer_vision is not None:
            for index, label_id in enumerate(detection_labels):
                label_name = computer_vision.get_label_name(label_id)
                resolved.append(label_name)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": int(label_id),
                        "position": None,
                    }
                )
        elif detection_labels is not None:
            for index, label_id in enumerate(detection_labels):
                label_name = str(label_id)
                resolved.append(label_name)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": int(label_id),
                        "position": None,
                    }
                )

        self.detections = resolved if resolved else None
        self.detection_candidates = candidates if candidates else None

    def route_request(
        self,
        transcription: str,
        pending_label: Optional[str] = None,
        detected_objects: Optional[list] = None,
        detection_labels: Optional[list[int]] = None,
    ) -> dict:
        text = str(transcription or "").strip().lower()
        if not text:
            return {"needs_vision": False, "mode": "repeat"}

        active_pending_label = pending_label if pending_label is not None else self.pending_label
        messages = self.formatter.build_route_messages(
            text,
            pending_label=active_pending_label,
            detected_objects=detected_objects,
            detection_labels=detection_labels,
        )
        if self.stream_json:
            thinking, content = self.chat_controller.chat_stream_json(
                messages=messages,
                schema=self.formatter.route_schema,
            )
        else:
            thinking, content = self.chat_controller.chat_non_stream_json(
                messages=messages,
                schema=self.formatter.route_schema,
            )
        if self.debug and thinking.strip():
            print("[SLM_DEBUG] Router thinking:")
            print(thinking)
        if self.debug and content.strip():
            print("[SLM_DEBUG] Router answer:")
            print(content)

        return self.formatter.parse_route_response(content)

    def respond_from_mode_hint(self, mode: str, transcription: str, detections: Optional[list[str]] = None) -> dict:
        detections = detections or []
        messages = self.formatter.build_mode_hint_messages(
            mode=mode,
            transcription=transcription,
            detections=detections,
        )

        try:
            text = self.chat_controller.chat_non_stream_text(messages=messages, max_tokens=100)
            feedback_source = "slm_feedback"
        except Exception as exc:
            self._debug_log(f"mode response fallback due to error: {exc}")
            text = ""
            feedback_source = "fallback_feedback"

        if not text:
            text = self.formatter.fallback_mode_response(mode=mode, detections=detections)
            feedback_source = "fallback_feedback"

        if mode == "stop":
            self.reset_task_context()

        self.pending_label = None
        return {
            "response": text,
            "mode": mode,
            "selected_label": None,
            "selected_label_id": None,
            "selected_candidate_index": None,
            "feedback_source": feedback_source,
        }

    def run_inference(self) -> dict:
        transcription = self.transcription
        detections = self.detections or []
        candidates = self.detection_candidates or []

        if not transcription:
            return {
                "response": "I didn't hear a command. Could you repeat?",
                "mode": "repeat",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
                "feedback_source": "fallback_feedback",
            }

        active_candidates = self.formatter.select_active_candidates(
            transcription=transcription,
            candidates=candidates,
            pending_label=self.pending_label,
        )

        prompt = self.formatter.build_prompt(
            transcription=transcription,
            detections=detections,
            candidates=active_candidates,
            pending_label=self.pending_label,
        )
        self._debug_log(
            f"prompt_build transcription={repr(transcription[:80])} detections={len(detections)} candidates={len(candidates)}"
        )

        self.history.append({"role": "user", "content": prompt})

        try:
            thinking = ""
            content = ""

            if self.prefer_tensorrt and self.tensorrt_manager.ready and not self.stream_json:
                try:
                    content = self.tensorrt_manager.chat(
                        messages=self.history,
                        max_new_tokens=self.num_predict,
                        temperature=self.temperature,
                    ) or ""
                    self._debug_log(f"tensorrt_response_len={len(content)}")
                except Exception as exc:
                    self._debug_log(f"tensorrt_error={exc}")
                    raise

            if not content.strip():
                if self.stream_json:
                    thinking, content = self.chat_controller.chat_stream_json(
                        messages=self.history,
                        schema=self.formatter.response_schema,
                    )
                else:
                    thinking, content = self.chat_controller.chat_non_stream_json(
                        messages=self.history,
                        schema=self.formatter.response_schema,
                    )
                    if self.debug and thinking.strip():
                        print("[SLM_DEBUG] Thinking:")
                        print(thinking)
                    if self.debug and content.strip():
                        print("[SLM_DEBUG] Answer:")
                        print(content)

            if self.debug and (thinking or content):
                print("")

            self._debug_log(f"response_len={len(content)}")

            if not content.strip():
                self._debug_log("empty stream content, retrying non-stream without thinking")
                _, retry_content = self.chat_controller.chat_non_stream_json(
                    messages=self.history,
                    schema=self.formatter.response_schema,
                )
                content = retry_content or content

            if not content.strip():
                self._debug_log("WARNING: empty response from model")
                return {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "clarification",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                    "feedback_source": "fallback_feedback",
                }

            assistant_message = {"role": "assistant", "content": content}
            if thinking.strip():
                assistant_message["thinking"] = thinking
            self.history.append(assistant_message)

            parsed = self.formatter.parse_response(content)
            parsed = self.formatter.sanitize_output(
                parsed=parsed,
                transcription=transcription,
                candidates=active_candidates,
                pending_label=self.pending_label,
            )
            parsed["feedback_source"] = "slm_feedback"

            if parsed["mode"] == "stop":
                self.reset_task_context()

            if parsed["mode"] == "clarification":
                requested_label = self.formatter.find_requested_label(transcription, active_candidates)
                self.pending_label = requested_label or self.pending_label
            else:
                self.pending_label = None

            return parsed

        except json.JSONDecodeError as err:
            self._debug_log(f"JSON parse error: {err}")
            return {
                "response": "I had trouble processing that. Could you repeat?",
                "mode": "clarification",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
                "feedback_source": "fallback_feedback",
            }
        except Exception as err:
            self._debug_log(f"inference error: {err}")
            raise

    def load_model(self) -> None:
        if self.prefer_tensorrt and self.tensorrt_manager.ready:
            try:
                self.tensorrt_manager.warmup()
                self._debug_log("TensorRT backend is ready")
                return
            except Exception as exc:
                self._debug_log(f"TensorRT warmup failed: {exc}")
                raise RuntimeError(f"TensorRT warmup failed: {exc}") from exc

        try:
            self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "{}"}],
                options={"num_predict": 2},
            )
            self._debug_log(f"Model {self.model_name} is ready")
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_name}' not found or not ready. "
                f"Create it with: ollama create {self.model_name} -f Modelfile"
            ) from exc

    def preload(self) -> None:
        self.load_model()


if __name__ == "__main__":
    os.environ["SLM_DEBUG"] = "1"
    mgr = SLM_Manager()
    mgr.load_model()
    mgr.set_transcription("give me a bottle")
    mgr.set_detections(detection_labels=[0, 0, 1])
    result = mgr.run_inference()
    print(json.dumps(result, indent=2))
