from __future__ import annotations

import json
from typing import Optional


class SLM_Formatter:
    """Builds prompts and validates/normalizes model JSON payloads."""

    def __init__(self, response_schema: dict):
        self.response_schema = response_schema

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
        return (
            f"Transcription: {transcription}\\n"
            f"Detected objects: {detections}\\n"
            f"Pending label from prior clarification (if any): {pending_label_text}\\n"
            f"Candidates (format: index, label, label_id, position [x,y,z]): {candidates_json}\\n"
            "Important: when transcription references prior clarification (e.g., 'left one', 'closest one'), "
            "resolve using candidates positions and return confirmation with selected_candidate_index.\\n"
            "Important: if pending label exists and user says pronouns like 'left one', resolve only within that label group.\\n"
            f"Return JSON matching this schema exactly: {schema_json}"
        )

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
            else:
                if requested_label:
                    count = sum(
                        1 for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                    )
                    if count > 1:
                        parsed["response"] = (
                            f"I found {count} {self._pluralize(requested_label, count)}. "
                            "Which one would you like? (e.g., left, right, closest, farthest?)"
                        )
                        parsed["mode"] = "clarification"
                    elif count == 1:
                        only = next(
                            c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                        )
                        parsed["response"] = f"I found one {requested_label}. I can pick it now."
                        parsed["mode"] = "confirmation"
                        parsed["selected_candidate_index"] = only.get("index")
                        parsed["selected_label"] = only.get("label")
                        parsed["selected_label_id"] = only.get("label_id")
                    else:
                        parsed["response"] = "I need more details to identify the object."
                        parsed["mode"] = "clarification"
                else:
                    parsed["response"] = "I need more details to identify the object."
                    parsed["mode"] = "clarification"

        if parsed.get("mode") == "clarification" and (
            "don't see" in normalized or "do not see" in normalized
        ) and requested_label:
            count = sum(
                1 for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
            )
            if count > 1:
                parsed["response"] = (
                    f"I found {count} {self._pluralize(requested_label, count)}. "
                    "Which one would you like? (e.g., left, right, closest, farthest?)"
                )
            elif count == 1:
                only = next(
                    c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                )
                parsed["response"] = f"I found one {requested_label}. I can pick it now."
                parsed["mode"] = "confirmation"
                parsed["selected_candidate_index"] = only.get("index")
                parsed["selected_label"] = only.get("label")
                parsed["selected_label_id"] = only.get("label_id")
            else:
                parsed["mode"] = "reformulate"
                parsed["response"] = "I don't see that object. Could you describe it differently?"

        if parsed.get("mode") == "confirmation" and parsed.get("selected_candidate_index") is None:
            if requested_label:
                same = [
                    c for c in candidates if str(c.get("label") or "").strip().lower() == requested_label
                ]
                if len(same) == 1:
                    parsed["selected_candidate_index"] = same[0].get("index")
                    parsed["selected_label"] = same[0].get("label")
                    parsed["selected_label_id"] = same[0].get("label_id")
                elif len(same) > 1:
                    parsed["mode"] = "clarification"
                    parsed["response"] = (
                        f"I found {len(same)} {self._pluralize(requested_label, len(same))}. "
                        "Which one would you like? (e.g., left, right, closest, farthest?)"
                    )
                    parsed["selected_label"] = None
                    parsed["selected_label_id"] = None

        return parsed
