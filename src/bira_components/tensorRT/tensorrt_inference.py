"""TensorRT inference wrapper compatible with the current `SLM_Manager`."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ENGINE_DIR = MODULE_DIR / "tensorrt_models" / "engines"


class TensorRTInferenceEngine:
    """Generic TensorRT chat wrapper driven by a local runner command."""

    def __init__(self, engine_dir: Optional[str] = None, device_id: int = 0):
        self.engine_dir = Path(engine_dir) if engine_dir else DEFAULT_ENGINE_DIR
        self.device_id = device_id
        self.config_path: Optional[Path] = None
        self.config = self._load_config()
        self.is_loaded = False
        self.last_error: Optional[str] = None
        self.runner_command: List[str] = []

    def _candidate_config_paths(self) -> List[Path]:
        config_override = os.getenv("TENSORRT_CONFIG_PATH")
        if config_override:
            return [Path(config_override).expanduser()]

        # Accept both runtime config.json and template tensorrt_config.json.
        candidates = [
            self.engine_dir / "config.json",
            self.engine_dir / "tensorrt_config.json",
            self.engine_dir / "engines" / "config.json",
            self.engine_dir / "tensorrt_models" / "engines" / "config.json",
            MODULE_DIR / "tensorrt_config.json",
        ]
        deduped: List[Path] = []
        seen = set()
        for path in candidates:
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(path)
        return deduped

    @staticmethod
    def _resolve_engine_dir_from_config(config: Dict[str, Any], config_path: Path) -> Optional[Path]:
        paths_cfg = config.get("paths")
        if not isinstance(paths_cfg, dict):
            return None

        configured = paths_cfg.get("engine_dir")
        if not isinstance(configured, str) or not configured.strip():
            return None

        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        return candidate

    def _load_config(self) -> Dict[str, Any]:
        candidates = self._candidate_config_paths()
        config_path = next((path for path in candidates if path.exists()), None)
        if config_path is None:
            self.config_path = candidates[0] if candidates else None
            searched = ", ".join(str(path) for path in candidates)
            if os.getenv("TENSORRT_RUNNER_COMMAND"):
                logger.info(
                    "TensorRT config not found. Tried: %s. Using TENSORRT_RUNNER_COMMAND from environment.",
                    searched,
                )
            else:
                logger.warning(
                    "TensorRT config not found. Tried: %s. Create config.json or set TENSORRT_RUNNER_COMMAND.",
                    searched,
                )
            return {}

        self.config_path = config_path
        if config_path.name == "tensorrt_config.json":
            logger.info("Using TensorRT template config at %s", config_path)

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
        except json.JSONDecodeError as exc:
            logger.error("TensorRT config is not valid JSON at %s (%s)", config_path, exc)
            return {}

        resolved_engine_dir = self._resolve_engine_dir_from_config(config, config_path)
        if resolved_engine_dir:
            self.engine_dir = resolved_engine_dir

        return config

    def _resolve_runner_command(self) -> List[str]:
        runtime_cfg = self.config.get("runtime", {})
        configured = os.getenv("TENSORRT_RUNNER_COMMAND") or runtime_cfg.get("runner_command")

        if isinstance(configured, list):
            return [str(part) for part in configured]
        if isinstance(configured, str) and configured.strip():
            return shlex.split(configured)
        return []

    def load_engine(self) -> bool:
        """Validate the TensorRT assets and the runner command."""
        if not self.engine_dir.exists() or not self.engine_dir.is_dir():
            self.last_error = (
                f"TensorRT engine directory not found: {self.engine_dir}. "
                "Set TENSORRT_ENGINE_DIR or provide tensorrt_engine_dir."
            )
            logger.warning(self.last_error)
            return False
        
        self.runner_command = self._resolve_runner_command()
        if not self.runner_command:
            self.last_error = (
                "No TensorRT runner command configured. "
                "Set runtime.runner_command in config.json or TENSORRT_RUNNER_COMMAND."
            )
            logger.warning(self.last_error)
            return False

        self.last_error = None
        self.is_loaded = True
        logger.info("TensorRT engine is ready via runner: %s", " ".join(self.runner_command))
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 192,
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """Generate text by delegating to the configured TensorRT runner."""
        if not self.is_loaded and not self.load_engine():
            raise RuntimeError("TensorRT engine is not ready.")

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "engine_dir": str(self.engine_dir),
            "device_id": self.device_id,
        }

        start_time = time.time()
        result = subprocess.run(
            self.runner_command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown TensorRT runner error"
            raise RuntimeError(stderr)

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("TensorRT runner returned no output.")

        generated_text = self._extract_generated_text(stdout)
        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "inference_time": elapsed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "engine": "TensorRT",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 192,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Expose an Ollama-like chat interface for `SLM_Manager`."""
        prompt = self._format_chat_prompt(messages)
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {
            "message": {
                "role": "assistant",
                "content": result["generated_text"],
            },
            "total_duration": int(result["inference_time"] * 1e9),
            "load_duration": 0,
            "eval_count": max_new_tokens,
            "engine": "TensorRT",
        }

    def _extract_generated_text(self, stdout: str) -> str:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout

        if isinstance(parsed, dict):
            if "generated_text" in parsed:
                return str(parsed["generated_text"])
            if "response" in parsed:
                return str(parsed["response"])
            if "message" in parsed and isinstance(parsed["message"], dict):
                return str(parsed["message"].get("content", ""))

        raise RuntimeError("TensorRT runner output must include generated_text, response, or message.content.")

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "config": self.config,
            "device_id": self.device_id,
            "engine_dir": str(self.engine_dir),
            "runner_command": self.runner_command,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TensorRTInferenceEngine()
    if not engine.load_engine():
        raise SystemExit("TensorRT engine is not ready.")

    response = engine.chat(
        messages=[{"role": "user", "content": "Can you give me the banana in front of you?"}],
        temperature=0.3,
    )
    print(response["message"]["content"])
