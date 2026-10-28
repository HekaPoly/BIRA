"""Simple benchmark script for Ollama vs TensorRT chat backends."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, List, Optional

from ollama import Client

from bira_components.tensorRT import TensorRTInferenceEngine

logger = logging.getLogger(__name__)


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    return str(getattr(getattr(response, "message", None), "content", ""))


class Benchmark:
    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}

    def benchmark_ollama(
        self,
        model_name: str = "qwen3:1.7b",
        test_prompts: Optional[List[str]] = None,
        num_runs: int = 3,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        prompts = test_prompts or [
            "Can you give me the banana?",
            "Pick up the red car and avoid the obstacles.",
            "What objects do you see?",
        ]
        client = Client(host="http://localhost:11434")
        timings = []

        for prompt in prompts:
            for run in range(num_runs):
                start_time = time.time()
                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                )
                elapsed = time.time() - start_time
                timings.append(
                    {
                        "prompt": prompt,
                        "run": run + 1,
                        "time": elapsed,
                        "tokens": len(_extract_content(response).split()),
                    }
                )

        stats = self._compute_stats("Ollama", model_name, timings)
        self.results["ollama"] = stats
        return stats

    def benchmark_tensorrt(
        self,
        engine_dir: Optional[str] = None,
        test_prompts: Optional[List[str]] = None,
        num_runs: int = 3,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        prompts = test_prompts or [
            "Can you give me the banana?",
            "Pick up the red car and avoid the obstacles.",
            "What objects do you see?",
        ]
        engine = TensorRTInferenceEngine(engine_dir=engine_dir)
        if not engine.load_engine():
            raise RuntimeError("TensorRT engine is not ready. Configure runtime.runner_command first.")

        timings = []
        for prompt in prompts:
            for run in range(num_runs):
                start_time = time.time()
                response = engine.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                elapsed = time.time() - start_time
                timings.append(
                    {
                        "prompt": prompt,
                        "run": run + 1,
                        "time": elapsed,
                        "tokens": len(_extract_content(response).split()),
                    }
                )

        stats = self._compute_stats(
            "TensorRT",
            engine.config.get("model_config", {}).get("model_name", "unknown"),
            timings,
        )
        self.results["tensorrt"] = stats
        return stats

    def _compute_stats(self, engine_name: str, model_name: str, timings: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_time = sum(item["time"] for item in timings)
        total_tokens = sum(item["tokens"] for item in timings)
        avg_latency = total_time / len(timings) if timings else 0.0
        return {
            "engine": engine_name,
            "model": model_name,
            "num_tests": len(timings),
            "avg_latency": avg_latency,
            "tokens_per_sec": (total_tokens / total_time) if total_time > 0 else 0.0,
            "timings": timings,
        }

    def compare_results(self) -> None:
        ollama = self.results.get("ollama")
        tensorrt = self.results.get("tensorrt")
        if ollama:
            print(f"Ollama {ollama['model']}: {ollama['avg_latency']:.3f}s avg, {ollama['tokens_per_sec']:.2f} tok/s")
        if tensorrt:
            print(f"TensorRT {tensorrt['model']}: {tensorrt['avg_latency']:.3f}s avg, {tensorrt['tokens_per_sec']:.2f} tok/s")
        if ollama and tensorrt and tensorrt["avg_latency"] > 0:
            print(f"Speedup: {ollama['avg_latency'] / tensorrt['avg_latency']:.2f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Benchmark Ollama against TensorRT.")
    parser.add_argument("--model", default="qwen3:1.7b", help="Ollama model name")
    parser.add_argument("--engine-dir", default=None, help="TensorRT engine directory")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt")
    args = parser.parse_args()

    benchmark = Benchmark()
    prompts = [
        "Can you give me the banana in front of you?",
        "Pick up the red car and avoid the obstacles.",
        "What do you see on the table?",
    ]

    try:
        benchmark.benchmark_ollama(model_name=args.model, test_prompts=prompts, num_runs=args.runs)
    except Exception as exc:
        logger.error("Ollama benchmark failed: %s", exc)

    try:
        benchmark.benchmark_tensorrt(engine_dir=args.engine_dir, test_prompts=prompts, num_runs=args.runs)
    except Exception as exc:
        logger.error("TensorRT benchmark failed: %s", exc)

    benchmark.compare_results()
