from __future__ import annotations

import json
from typing import Optional


class SLM_Formatter:
    """Builds prompts and validates/normalizes model JSON payloads."""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": [
                    "confirmation",
                    "clarification",
                    "reformulate",
                    "repeat",
                    "unclear_action",
                    "conversing",
                    "out_of_scope",
                    "inappropriate",
                    "stop",
                ],
            },
            "selected_candidate_index": {"type": ["integer", "null"]},
            "selected_label": {"type": ["string", "null"]},
            "selected_label_id": {"type": ["integer", "null"]},
        },
        "required": [
            "response",
            "mode",
            "selected_candidate_index",
            "selected_label",
            "selected_label_id",
        ],
        "additionalProperties": False,
    }

    ROUTE_SCHEMA = {
        "type": "object",
        "properties": {
            "needs_vision": {"type": "boolean"},
            "mode": {
                "type": "string",
                "enum": [
                    "confirmation",
                    "clarification",
                    "reformulate",
                    "repeat",
                    "unclear_action",
                    "conversing",
                    "out_of_scope",
                    "inappropriate",
                    "stop",
                ],
            },
        },
        "required": ["needs_vision", "mode"],
        "additionalProperties": False,
    }

    def __init__(self, response_schema: dict):
        self.response_schema = response_schema or self.RESPONSE_SCHEMA

    @property
    def route_schema(self) -> dict:
        return self.ROUTE_SCHEMA

    @staticmethod
    def _parse_first_json(text: str) -> dict:
        text = text.strip()
        object_start = text.find("{")
        array_start = text.find("[")

        if object_start == -1 and array_start == -1:
            raise json.JSONDecodeError("No JSON value found", text, 0)

        start = (
            object_start
            if array_start == -1
            else array_start
            if object_start == -1
            else min(object_start, array_start)
        )

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

    def build_prompt(
        self,
        transcription: str,
        detections: list[str],
        candidates: Optional[list[dict]],
        pending_label: Optional[str],
    ) -> str:
        candidates_json = json.dumps(candidates or [], ensure_ascii=False)
        schema_json = json.dumps(self.response_schema, ensure_ascii=False)
        pending_label_text = pending_label or "null"
        vision_status = "available" if (detections or candidates) else "unavailable"
        return (
            f"Transcription: {transcription}\\n"
            f"Vision status for this turn: {vision_status}\\n"
            f"Detected objects: {detections}\\n"
            f"Pending label from prior clarification (if any): {pending_label_text}\\n"
            f"Candidates (format: index, label, label_id, position [x,y,z]): {candidates_json}\\n"
            "Important: if vision status is unavailable, do not claim to see objects. "
            "Use conversing/out_of_scope/repeat/unclear_action based on transcription intent.\\n"
            "Important: when transcription references prior clarification (e.g., 'left one', 'closest one'), "
            "resolve using candidates positions and return confirmation with selected_candidate_index.\\n"
            "Important: if pending label exists and user says pronouns like 'left one', resolve only within that label group.\\n"
            f"Return JSON matching this schema exactly: {schema_json}"
        )

    def build_route_messages(self, transcription: str) -> list[dict]:
        route_system_prompt = (
            "You are a routing classifier for a robotic arm assistant. "
            "Your ONLY job: decide if the request needs VISION (camera input) before planning. "
            "Return strict JSON only: {\"needs_vision\": boolean, \"mode\": string}. "
            "\n"
            "RULES:\n"
            "1. needs_vision=TRUE if: user asks for an object (pick, grab, bring, show, give, hold, take) with any noun. "
            "   Even 'give me a cup' needs vision to identify which cup.\n"
            "2. needs_vision=FALSE if: user stops, chats, asks out-of-scope (run, walk, cook), or unclear intent.\n"
            "3. If needs_vision=TRUE, mode defaults to 'clarification' (planning will handle object matching).\n"
            "4. If needs_vision=FALSE, mode reflects intent: 'stop', 'conversing', 'out_of_scope', 'unclear_action', 'repeat'."
        )
        return [
            {"role": "system", "content": route_system_prompt},
            {"role": "user", "content": f"Transcription: {transcription}"},
        ]

    def parse_route_response(self, raw_response: str) -> dict:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = self._parse_first_json(raw_response)

        if not isinstance(parsed, dict):
            raise ValueError("Route response is not JSON object")

        mode = str(parsed.get("mode", "clarification")).strip().lower()
        if mode not in {
            "confirmation",
            "clarification",
            "reformulate",
            "repeat",
            "unclear_action",
            "conversing",
            "out_of_scope",
            "inappropriate",
            "stop",
        }:
            mode = "clarification"

        needs_vision = parsed.get("needs_vision", True)
        if not isinstance(needs_vision, bool):
            needs_vision = str(needs_vision).strip().lower() in {"1", "true", "yes", "on"}

        return {"needs_vision": needs_vision, "mode": mode}

    def build_mode_hint_messages(self, mode: str, transcription: str, detections: Optional[list[str]]) -> list[dict]:
        system_prompt = (
            "You are a concise robotic-arm assistant. "
            "Write exactly one short, natural sentence for the user based on the provided mode."
        )
        user_prompt = (
            f"Mode: {mode}\n"
            f"User transcription: {transcription}\n"
            f"Detected objects: {detections or []}\n"
            "Rules: never output JSON, never output mode name, be polite and practical."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def fallback_mode_response(mode: str, detections: Optional[list[str]] = None) -> str:
        detections = detections or []
        if mode == "conversing":
            return "I'm doing well, thanks for asking!"
        if mode == "out_of_scope":
            seen = ", ".join(detections) if detections else "no objects right now"
            return (
                "I'm sorry, I can't do that action. "
                f"I'm a robotic arm and can pick, grab, or bring objects. I can see: {seen}."
            )
        if mode == "repeat":
            return "I didn't catch that. Could you repeat?"
        if mode == "stop":
            return "Alright, cancelling."
        if mode == "inappropriate":
            return "I can't help with that. Please ask something safe."
        if mode == "unclear_action":
            return "I couldn't understand the action. Could you ask me to pick, grab, or bring an object?"
        return "Could you rephrase your request?"

    def parse_response(self, raw_response: str) -> dict:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = self._parse_first_json(raw_response)

        if isinstance(parsed, list) and parsed:
            parsed = parsed[-1]

        if not isinstance(parsed, dict):
            raise ValueError("Response is not JSON object")

        mode = str(parsed.get("mode", "clarification")).strip().lower()
        if mode not in {
            "confirmation",
            "clarification",
            "reformulate",
            "repeat",
            "unclear_action",
            "conversing",
            "out_of_scope",
            "inappropriate",
            "stop",
        }:
            mode = "clarification"

        response_text = str(parsed.get("response", "I didn't understand. Could you repeat?"))

        selected_label = parsed.get("selected_label")
        selected_label = (str(selected_label).strip() or None) if selected_label is not None else None

        selected_label_id = parsed.get("selected_label_id")
        try:
            selected_label_id = int(selected_label_id) if selected_label_id is not None else None
        except (TypeError, ValueError):
            selected_label_id = None

        selected_candidate_index = parsed.get("selected_candidate_index")
        try:
            selected_candidate_index = int(selected_candidate_index) if selected_candidate_index is not None else None
        except (TypeError, ValueError):
            selected_candidate_index = None

        return {
            "response": response_text,
            "mode": mode,
            "selected_label": selected_label,
            "selected_label_id": selected_label_id,
            "selected_candidate_index": selected_candidate_index,
        }

    @staticmethod
    def _pluralize(label: str, count: int) -> str:
        if count == 1:
            return label
        if label.endswith("s"):
            return label
        return f"{label}s"

    def find_requested_label(self, transcription: str, candidates: list[dict]) -> Optional[str]:
        text = str(transcription or "").strip().lower()
        if not text or not candidates:
            return None

        label_counts: dict[str, int] = {}
        for candidate in candidates:
            label = str(candidate.get("label") or "").strip().lower()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        for label in label_counts:
            if label in text:
                return label

        return None

    @staticmethod
    def is_disambiguation_followup(transcription: str) -> bool:
        text = str(transcription or "").strip().lower()
        if not text:
            return False
        hints = ["left", "right", "top", "bottom", "closest", "farthest", "one", "this", "that"]
        return any(hint in text for hint in hints)

    def _resolve_label_match(
        self, label: str, candidates: list[dict], empty_response: str = "I need more details to identify the object."
    ) -> tuple[str, str, Optional[int], Optional[str], Optional[int]]:
        """
        Helper: Find candidates matching label and generate appropriate response.
        Returns: (response_text, mode, selected_candidate_index, selected_label, selected_label_id)
        """
        matching = [c for c in candidates if str(c.get("label") or "").strip().lower() == label]
        count = len(matching)

        if count == 0:
            return empty_response, "clarification", None, None, None

        if count == 1:
            only = matching[0]
            response = f"I found one {label}. I can pick it now."
            return response, "confirmation", only.get("index"), only.get("label"), only.get("label_id")

        # count > 1
        response = (
            f"I found {count} {self._pluralize(label, count)}. "
            "Which one would you like? (e.g., left, right, closest, farthest?)"
        )
        return response, "clarification", None, None, None

    def select_active_candidates(
        self,
        transcription: str,
        candidates: list[dict],
        pending_label: Optional[str],
    ) -> list[dict]:
        if not candidates:
            return []

        explicit_label = self.find_requested_label(transcription, candidates)
        if explicit_label:
            return [c for c in candidates if str(c.get("label") or "").strip().lower() == explicit_label]

        if pending_label and self.is_disambiguation_followup(transcription):
            narrowed = [c for c in candidates if str(c.get("label") or "").strip().lower() == pending_label]
            if narrowed:
                return narrowed

        return candidates

    def sanitize_output(
        self,
        parsed: dict,
        transcription: str,
        candidates: list[dict],
        pending_label: Optional[str],
    ) -> dict:
        response_text = str(parsed.get("response") or "").strip()
        normalized = response_text.lower()
        requested_label = self.find_requested_label(transcription, candidates) or pending_label

        if not response_text or normalized in {"...", "…"}:
            if parsed.get("mode") == "confirmation":
                parsed["response"] = "Understood. I will pick that object."
            elif requested_label:
                response, mode, cand_idx, label, label_id = self._resolve_label_match(
                    requested_label, candidates
                )
                parsed["response"] = response
                parsed["mode"] = mode
                if cand_idx is not None:
                    parsed["selected_candidate_index"] = cand_idx
                    parsed["selected_label"] = label
                    parsed["selected_label_id"] = label_id
            else:
                parsed["response"] = "I need more details to identify the object."
                parsed["mode"] = "clarification"

        if parsed.get("mode") == "clarification" and (
            "don't see" in normalized or "do not see" in normalized
        ) and requested_label:
            response, mode, cand_idx, label, label_id = self._resolve_label_match(
                requested_label, candidates, empty_response="I don't see that object. Could you describe it differently?"
            )
            if response.startswith("I found"):
                parsed["response"] = response
                parsed["mode"] = mode
            else:
                parsed["mode"] = "reformulate"
                parsed["response"] = response

        if parsed.get("mode") == "confirmation" and parsed.get("selected_candidate_index") is None:
            if requested_label:
                response, mode, cand_idx, label, label_id = self._resolve_label_match(
                    requested_label, candidates
                )
                if cand_idx is not None:
                    parsed["selected_candidate_index"] = cand_idx
                    parsed["selected_label"] = label
                    parsed["selected_label_id"] = label_id
                else:
                    parsed["mode"] = "clarification"
                    parsed["response"] = response

        return parsed
