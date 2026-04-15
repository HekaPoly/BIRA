from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from bira_components import history as bira_history
from cv_viewer import labels
from ollama import Client

try:
    from bira_components.tensorRT import TensorRTInferenceEngine
except Exception:
    TensorRTInferenceEngine = None


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean flag from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_first_json(text: str):
    """Extract the first JSON object or array embedded in a string."""
    text = text.strip()
    object_start = text.find("{")
    array_start = text.find("[")

    if object_start == -1 and array_start == -1:
        raise json.JSONDecodeError("No JSON value found", text, 0)

    if object_start == -1:
        start = array_start
    elif array_start == -1:
        start = object_start
    else:
        start = min(object_start, array_start)

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise json.JSONDecodeError("Unbalanced JSON", text, start)


class SLM_Manager:
    """Bridge between BIRA and the language backend."""

    def __init__(
        self,
        model_name: str = "bira-assistant",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.3,
        mode: str = "local",
        api_key: Optional[str] = None,
        prefer_tensorrt: Optional[bool] = None,
        tensorrt_engine_dir: Optional[str] = None,
        tensorrt_fallback_to_ollama: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else int(os.getenv("SLM_NUM_PREDICT", "900"))
        )
        self.temperature = temperature
        self.mode = mode
        self.debug = _env_flag("SLM_DEBUG", default=False)
        self.history: list[dict] = []
        self.detections: Optional[list[str]] = None
        self.detection_candidates: Optional[list[dict]] = None
        self.transcription: Optional[str] = None
        self.pending_label: Optional[str] = None
        self.last_backend: Optional[str] = None
        self.prefer_tensorrt = (
            _env_flag("USE_TENSORRT", default=True)
            if prefer_tensorrt is None
            else prefer_tensorrt
        )
        self.tensorrt_fallback_to_ollama = (
            _env_flag("TENSORRT_FALLBACK_TO_OLLAMA", default=True)
            if tensorrt_fallback_to_ollama is None
            else tensorrt_fallback_to_ollama
        )
        default_engine_dir = Path(__file__).resolve().parent / "tensorRT" / "tensorrt_models" / "engines"
        env_engine_dir = os.getenv("TENSORRT_ENGINE_DIR")
        resolved_engine_dir = tensorrt_engine_dir or env_engine_dir or default_engine_dir
        self.tensorrt_engine_dir = Path(resolved_engine_dir)
        self.trt_engine: Optional[TensorRTInferenceEngine] = None
        self.trt_ready = False
        self.model_loaded = False
        self.response_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["confirmation", "clarification", "repeat", "stop"],
                },
                "request_scope": {
                    "type": "string",
                    "enum": ["in_scope", "out_of_scope"],
                },
                "selected_candidate_index": {"type": ["integer", "null"]},
                "selected_label": {"type": ["string", "null"]},
                "selected_label_id": {"type": ["integer", "null"]},
            },
            "required": [
                "response",
                "mode",
                "request_scope",
                "selected_candidate_index",
                "selected_label",
                "selected_label_id",
            ],
            "additionalProperties": False,
        }

        if mode == "cloud":
            if not api_key:
                raise ValueError("API key required for cloud mode")
            self.client = Client(
                host="https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            self.client = Client(host="http://localhost:11434")

        if self.mode != "local" and self.prefer_tensorrt:
            print("TensorRT is only available in local mode. Falling back to Ollama.")
            bira_history.log_event("tensorrt_disabled", component="slm_manager", reason="non_local_mode")
            self.prefer_tensorrt = False

        if self.prefer_tensorrt:
            if TensorRTInferenceEngine is None:
                print("TensorRT support is unavailable. Falling back to Ollama.")
                bira_history.log_event("tensorrt_unavailable", component="slm_manager", reason="import_failed")
                self.prefer_tensorrt = False
            else:
                self._init_tensorrt_engine()

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def _init_tensorrt_engine(self) -> None:
        bira_history.log_event(
            "tensorrt_load_started",
            component="slm_manager",
            engine_dir=str(self.tensorrt_engine_dir),
        )
        try:
            self.trt_engine = TensorRTInferenceEngine(engine_dir=str(self.tensorrt_engine_dir))
            self.trt_ready = bool(self.trt_engine.load_engine())
            if self.trt_ready:
                bira_history.log_event(
                    "tensorrt_load_succeeded",
                    component="slm_manager",
                    engine_dir=str(self.tensorrt_engine_dir),
                )
                print(f"TensorRT ready (engine: {self.tensorrt_engine_dir})")
                return

            details = getattr(self.trt_engine, "last_error", None)
            self.trt_engine = None
            bira_history.log_event(
                "tensorrt_load_unavailable",
                component="slm_manager",
                engine_dir=str(self.tensorrt_engine_dir),
            )
            if not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(f"No TensorRT engine available in {self.tensorrt_engine_dir}")
            if details:
                print(
                    f"TensorRT engine not ready in {self.tensorrt_engine_dir} "
                    f"({details}). Falling back to Ollama."
                )
            else:
                print(f"TensorRT engine not ready in {self.tensorrt_engine_dir}. Falling back to Ollama.")
        except Exception as exc:
            self.trt_ready = False
            self.trt_engine = None
            bira_history.log_event(
                "tensorrt_load_failed",
                component="slm_manager",
                engine_dir=str(self.tensorrt_engine_dir),
                error=str(exc),
            )
            if not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(f"TensorRT initialization failed: {exc}") from exc
            print(f"TensorRT unavailable ({exc}). Falling back to Ollama.")

    def _warmup_tensorrt_backend(self) -> None:
        if not self.trt_ready or not self.trt_engine:
            return

        self.trt_engine.chat(
            messages=[{"role": "user", "content": "Reply with {}."}],
            max_new_tokens=8,
            temperature=0.0,
        )

    def _warmup_ollama_backend(self) -> None:
        self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": "{}"}],
            options={"num_predict": 2, "temperature": 0},
        )

    def reset_conversation(self) -> None:
        self.history = []
        self.detections = None
        self.detection_candidates = None
        self.transcription = None
        self.pending_label = None

    def set_transcription(self, transcription: str) -> None:
        self.transcription = transcription

    def set_detections(
        self,
        detection_labels: Optional[list[int]] = None,
        detected_objects: Optional[list] = None,
        computer_vision=None,
    ) -> None:
        resolved: list[str] = []
        candidates: list[dict] = []

        if detected_objects:
            for index, obj in enumerate(detected_objects):
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                label_id = int(raw)
                if computer_vision is not None:
                    label_name = computer_vision.get_label_name(label_id)
                else:
                    label_name = labels.labelDict.get(label_id, str(label_id))
                resolved.append(label_name)

                position = getattr(obj, "position", None)
                position_value = None
                if position is not None and len(position) >= 3:
                    position_value = [float(position[0]), float(position[1]), float(position[2])]

                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": label_id,
                        "position": position_value,
                    }
                )
        elif detection_labels:
            for index, label_id in enumerate(detection_labels):
                resolved_label_id = int(label_id)
                if computer_vision is not None:
                    label_name = computer_vision.get_label_name(resolved_label_id)
                else:
                    label_name = labels.labelDict.get(resolved_label_id, str(resolved_label_id))
                resolved.append(label_name)
                candidates.append(
                    {
                        "index": index,
                        "label": label_name,
                        "label_id": resolved_label_id,
                        "position": None,
                    }
                )

        self.detections = resolved if resolved else None
        self.detection_candidates = candidates if candidates else None

    def _build_prompt(self, transcription: str, detections: list[str], candidates: Optional[list[dict]]) -> str:
        candidates_json = json.dumps(candidates or [], ensure_ascii=False)
        schema_json = json.dumps(self.response_schema, ensure_ascii=False)
        pending_label_text = self.pending_label or "null"
        return (
            f"Transcription: {transcription}\n"
            f"Detected objects: {detections}\n"
            f"Pending label from prior clarification (if any): {pending_label_text}\n"
            f"Candidates (format: index, label, label_id, position [x,y,z]): {candidates_json}\n"
            "Important: when transcription references prior clarification (e.g., 'left one', 'closest one'), "
            "resolve using candidates positions and return confirmation with selected_candidate_index.\n"
            "Important: if pending label exists and user says pronouns like 'left one', resolve only within that label group.\n"
            f"Return JSON matching this schema exactly: {schema_json}"
        )

    @staticmethod
    def _as_dict(value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    def _parse_response(self, raw_response: str) -> dict:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = _parse_first_json(raw_response)

        if isinstance(parsed, list) and parsed:
            parsed = parsed[-1]

        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")

        mode = str(parsed.get("mode", "clarification")).strip().lower()
        if mode not in {"confirmation", "clarification", "repeat", "stop"}:
            mode = "clarification"

        request_scope = str(parsed.get("request_scope", "in_scope")).strip().lower()
        if request_scope not in {"in_scope", "out_of_scope"}:
            request_scope = "in_scope"

        response_text = str(parsed.get("response", "I didn't understand. Could you repeat?")).strip()

        selected_label = parsed.get("selected_label")
        selected_label = (str(selected_label).strip() or None) if selected_label is not None else None

        selected_label_id = parsed.get("selected_label_id")
        try:
            selected_label_id = int(selected_label_id) if selected_label_id is not None else None
        except (TypeError, ValueError):
            selected_label_id = None

        selected_candidate_index = parsed.get("selected_candidate_index")
        try:
            selected_candidate_index = (
                int(selected_candidate_index) if selected_candidate_index is not None else None
            )
        except (TypeError, ValueError):
            selected_candidate_index = None

        return {
            "response": response_text or "I didn't understand. Could you repeat?",
            "mode": mode,
            "request_scope": request_scope,
            "selected_label": selected_label,
            "selected_label_id": selected_label_id,
            "selected_candidate_index": selected_candidate_index,
        }

    @staticmethod
    def _pluralize(label: str, count: int) -> str:
        if count == 1 or label.endswith("s"):
            return label
        return f"{label}s"

    def _find_requested_label(self, transcription: str, candidates: list[dict]) -> Optional[str]:
        text = str(transcription or "").strip().lower()
        if not text or not candidates:
            return None

        labels_in_scene = {
            str(candidate.get("label") or "").strip().lower()
            for candidate in candidates
            if str(candidate.get("label") or "").strip()
        }
        for label_name in labels_in_scene:
            if label_name and label_name in text:
                return label_name
        return None

    @staticmethod
    def _is_disambiguation_followup(transcription: str) -> bool:
        text = str(transcription or "").strip().lower()
        hints = ["left", "right", "top", "bottom", "closest", "farthest", "one", "this", "that"]
        return any(hint in text for hint in hints)

    def _select_active_candidates(self, transcription: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        explicit_label = self._find_requested_label(transcription, candidates)
        if explicit_label:
            return [c for c in candidates if str(c.get("label") or "").strip().lower() == explicit_label]

        if self.pending_label and self._is_disambiguation_followup(transcription):
            narrowed = [
                c for c in candidates if str(c.get("label") or "").strip().lower() == self.pending_label
            ]
            if narrowed:
                return narrowed

        return candidates

    def _sanitize_output(self, parsed: dict, transcription: str, candidates: list[dict]) -> dict:
        response_text = str(parsed.get("response") or "").strip()
        requested_label = self._find_requested_label(transcription, candidates) or self.pending_label
        matching = []
        if requested_label:
            matching = [
                c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
            ]

        if not response_text or response_text in {"...", "…"}:
            if len(matching) == 1:
                only = matching[0]
                parsed["response"] = f"I found one {requested_label}. I can pick it now."
                parsed["mode"] = "confirmation"
                parsed["selected_candidate_index"] = only.get("index")
                parsed["selected_label"] = only.get("label")
                parsed["selected_label_id"] = only.get("label_id")
            elif len(matching) > 1:
                parsed["response"] = (
                    f"I found {len(matching)} {self._pluralize(requested_label, len(matching))}. "
                    "Which one would you like? (left, right, closest, farthest?)"
                )
                parsed["mode"] = "clarification"
                parsed["selected_candidate_index"] = None
                parsed["selected_label"] = None
                parsed["selected_label_id"] = None
            else:
                parsed["response"] = "I need more details to identify the object."
                parsed["mode"] = "clarification"

        if parsed.get("mode") == "confirmation" and parsed.get("selected_candidate_index") is None:
            if len(matching) == 1:
                only = matching[0]
                parsed["selected_candidate_index"] = only.get("index")
                parsed["selected_label"] = only.get("label")
                parsed["selected_label_id"] = only.get("label_id")
            elif len(matching) > 1:
                parsed["mode"] = "clarification"
                parsed["response"] = (
                    f"I found {len(matching)} {self._pluralize(requested_label, len(matching))}. "
                    "Which one would you like? (left, right, closest, farthest?)"
                )
                parsed["selected_candidate_index"] = None
                parsed["selected_label"] = None
                parsed["selected_label_id"] = None

        return parsed

    def _chat_non_stream(self) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=self.history,
            stream=False,
            format=self.response_schema,
            options={"temperature": self.temperature, "num_predict": self.max_new_tokens},
        )
        data = self._as_dict(response)
        message = self._as_dict(data.get("message", {}))
        return str(message.get("content") or "")

    def _chat_tensorrt(self) -> str:
        if not (self.prefer_tensorrt and self.trt_ready and self.trt_engine):
            return ""

        response = self.trt_engine.chat(
            messages=self.history,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        data = self._as_dict(response)
        message = self._as_dict(data.get("message", {}))
        return str(message.get("content") or "")

    def run_inference(self) -> dict:
        transcription = self.transcription
        detections = self.detections
        candidates = self.detection_candidates or []

        if not transcription:
            return {
                "response": "I didn't hear a command. Could you repeat?",
                "mode": "repeat",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        if not detections:
            return {
                "response": "I don't see any relevant object. Could you clarify?",
                "mode": "clarification",
                "request_scope": "in_scope",
                "selected_label": None,
                "selected_label_id": None,
                "selected_candidate_index": None,
            }

        active_candidates = self._select_active_candidates(transcription, candidates)
        prompt = self._build_prompt(transcription, detections, active_candidates)
        self.history.append({"role": "user", "content": prompt})
        self.last_backend = None

        try:
            content = ""
            if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
                try:
                    content = self._chat_tensorrt()
                    if content.strip():
                        self.last_backend = "TensorRT"
                except Exception as exc:
                    bira_history.log_event("tensorrt_inference_failed", component="slm_manager", error=str(exc))
                    if not self.tensorrt_fallback_to_ollama:
                        raise

            if not content.strip():
                content = self._chat_non_stream()
                self.last_backend = "Ollama"

            if not content.strip():
                return {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "clarification",
                    "request_scope": "in_scope",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }

            self.history.append({"role": "assistant", "content": content})

            try:
                parsed = self._parse_response(content)
            except (json.JSONDecodeError, ValueError) as err:
                bira_history.log_event("slm_response_parse_failed", component="slm_manager", error=str(err))
                return {
                    "response": "I didn't understand. Could you repeat?",
                    "mode": "clarification",
                    "request_scope": "in_scope",
                    "selected_label": None,
                    "selected_label_id": None,
                    "selected_candidate_index": None,
                }

            parsed = self._sanitize_output(parsed, transcription, active_candidates)

            if parsed["mode"] == "repeat" and transcription.strip() and detections:
                parsed["mode"] = "clarification"
                if "repeat" in parsed["response"].lower():
                    parsed["response"] = "I understood the request, but I need more details about the object."

            if parsed["mode"] == "stop":
                bira_history.log_event("conversation_reset", component="slm_manager", reason="stop_mode")
                self.reset_conversation()
            elif parsed["mode"] == "clarification":
                requested_label = self._find_requested_label(transcription, active_candidates)
                if requested_label:
                    self.pending_label = requested_label
            else:
                self.pending_label = None

            return parsed
        except Exception as err:
            self._debug_log(f"inference error: {err}")
            raise

    def load_model(self) -> None:
        if self.model_loaded:
            return

        tensorrt_ready = False
        if self.prefer_tensorrt:
            if self.trt_ready and self.trt_engine:
                self._warmup_tensorrt_backend()
                tensorrt_ready = True
                if not self.tensorrt_fallback_to_ollama:
                    self.model_loaded = True
                    return
            elif not self.tensorrt_fallback_to_ollama:
                raise RuntimeError(
                    "TensorRT was requested but no runnable engine is available. "
                    f"Expected assets in {self.tensorrt_engine_dir}."
                )

        try:
            self._warmup_ollama_backend()
        except Exception as exc:
            if tensorrt_ready:
                self.model_loaded = True
                return
            raise RuntimeError(
                f"Model '{self.model_name}' not found or not ready. "
                f"Create it with: ollama create {self.model_name} -f Modelfile"
            ) from exc

        self.model_loaded = True

    def preload(self) -> None:
        self.load_model()

    def destroy(self) -> None:
        return


if __name__ == "__main__":
    os.environ["SLM_DEBUG"] = "1"
    manager = SLM_Manager()
    manager.load_model()
    manager.set_transcription("give me the bottle")
    manager.set_detections(detection_labels=[39, 39, 41])
    print(json.dumps(manager.run_inference(), indent=2))
