from __future__ import annotations

from typing import Any


class SLM_Controller:
    """Encapsulates Ollama chat interactions (streaming, thinking, retries)."""

    def __init__(self, client, model_name: str, temperature: float, num_predict: int, debug: bool = False):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.debug = debug

    @staticmethod
    def _as_dict(value: Any) -> dict:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[SLM_DEBUG] {message}")

    def _chat(self, messages: list[dict], *, stream: bool, think: bool, schema: dict | None = None, max_tokens: int | None = None):
        options = {"temperature": self.temperature, "num_predict": max_tokens or self.num_predict}
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": options,
        }
        if schema is not None:
            kwargs["format"] = schema
        return self.client.chat(**kwargs)

    def chat_stream_json(self, messages: list[dict], schema: dict) -> tuple[str, str]:
        thinking = ""
        content = ""

        stream = self._chat(messages, stream=True, think=True, schema=schema)

        in_thinking = False
        for chunk in stream:
            chunk_data = self._as_dict(chunk)
            message = self._as_dict(chunk_data.get("message", {}))
            chunk_thinking = str(message.get("thinking") or "")
            chunk_content = str(message.get("content") or "")

            if chunk_thinking:
                if not in_thinking:
                    print("[SLM_THINKING]")
                    in_thinking = True
                print(chunk_thinking, end="", flush=True)
                thinking += chunk_thinking
            elif chunk_content:
                if in_thinking:
                    print("\n[SLM_ANSWER]")
                    in_thinking = False
                print(chunk_content, end="", flush=True)
                content += chunk_content

        if thinking or content:
            print("")

        self._debug_log(f"response_len={len(content)}")
        return thinking, content

    def chat_non_stream_json(self, messages: list[dict], schema: dict) -> tuple[str, str]:
        fallback = self._chat(messages, stream=False, think=True, schema=schema)
        data = self._as_dict(fallback)
        message = self._as_dict(data.get("message", {}))
        return str(message.get("thinking") or ""), str(message.get("content") or "")

    def chat_non_stream_text(self, messages: list[dict], max_tokens: int = 120) -> str:
        result = self._chat(messages, stream=False, think=False, max_tokens=max_tokens)
        data = self._as_dict(result)
        message = self._as_dict(data.get("message", {}))
        return str(message.get("content") or "").strip()
