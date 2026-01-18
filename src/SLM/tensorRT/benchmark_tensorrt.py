"""
------------------------------------------------------------------------------------
TensorRT Benchmark
------------------------------------------------------------------------------------
2026-01-07 v1.0 - Benchmark pour comparer Ollama vs TensorRT
------------------------------------------------------------------------------------
DESCRIPTION GÉNÉRALE
------------------------------------------------------------------------------------
Ce script compare les performances entre:
  - Ollama standard
  - TensorRT optimisé

Métriques mesurées:
  - Temps de latence (première génération)
  - Débit (tokens/seconde)
  - Utilisation mémoire GPU
  - Temps de chargement du modèle
"""

import time
import psutil
import logging
from typing import Dict, List, Any
from ollama import Client
from SLM.tensorRT.tensorrt_inference import TensorRTInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmark:
    """Classe pour benchmarker les performances."""
    
    def __init__(self):
        self.results = {
            "ollama": [],
            "tensorrt": []
        }
    
    def benchmark_ollama(
        self,
        model_name: str = "BIRA",
        test_prompts: List[str] = None,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark du client Ollama standard.
        
        Args:
            model_name: Nom du modèle Ollama
            test_prompts: Liste de prompts de test
            num_runs: Nombre d'exécutions par prompt
            
        Returns:
            Statistiques de performance
        """
        logger.info(f"🔍 Benchmark Ollama: {model_name}")
        
        if test_prompts is None:
            test_prompts = [
                "Can you give me the banana?",
                "Pick up the red car and avoid the blue bus.",
                "What objects do you see?"
            ]
        
        client = Client(host='http://localhost:11434')
        
        timings = []
        
        for prompt in test_prompts:
            for run in range(num_runs):
                messages = [{"role": "user", "content": prompt}]
                
                start_time = time.time()
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": 0.4}
                )
                elapsed = time.time() - start_time
                
                timings.append({
                    "prompt": prompt,
                    "run": run + 1,
                    "time": elapsed,
                    "tokens": len(response.message.content.split())
                })
                
                logger.info(f"  Run {run+1}: {elapsed:.3f}s")
        
        # Calculer les statistiques
        avg_time = sum(t["time"] for t in timings) / len(timings)
        total_tokens = sum(t["tokens"] for t in timings)
        total_time = sum(t["time"] for t in timings)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        stats = {
            "engine": "Ollama",
            "model": model_name,
            "num_tests": len(timings),
            "avg_latency": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "timings": timings
        }
        
        self.results["ollama"] = stats
        return stats
    
    def benchmark_tensorrt(
        self,
        test_prompts: List[str] = None,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark du moteur TensorRT.
        
        Args:
            test_prompts: Liste de prompts de test
            num_runs: Nombre d'exécutions par prompt
            
        Returns:
            Statistiques de performance
        """
        logger.info("🔍 Benchmark TensorRT")
        
        if test_prompts is None:
            test_prompts = [
                "Can you give me the banana?",
                "Pick up the red car and avoid the blue bus.",
                "What objects do you see?"
            ]
        
        engine = TensorRTInferenceEngine()
        
        # Charger l'engine
        load_start = time.time()
        if not engine.load_engine():
            logger.error("❌ Impossible de charger l'engine TensorRT")
            return {}
        load_time = time.time() - load_start
        
        timings = []
        
        for prompt in test_prompts:
            for run in range(num_runs):
                messages = [{"role": "user", "content": prompt}]
                
                start_time = time.time()
                response = engine.chat(
                    messages=messages,
                    temperature=0.4
                )
                elapsed = time.time() - start_time
                
                timings.append({
                    "prompt": prompt,
                    "run": run + 1,
                    "time": elapsed,
                    "tokens": len(response["message"]["content"].split())
                })
                
                logger.info(f"  Run {run+1}: {elapsed:.3f}s")
        
        # Calculer les statistiques
        avg_time = sum(t["time"] for t in timings) / len(timings)
        total_tokens = sum(t["tokens"] for t in timings)
        total_time = sum(t["time"] for t in timings)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        stats = {
            "engine": "TensorRT",
            "model": engine.config.get("model_name", "unknown"),
            "load_time": load_time,
            "num_tests": len(timings),
            "avg_latency": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "timings": timings
        }
        
        self.results["tensorrt"] = stats
        return stats
    
    def compare_results(self) -> None:
        """Affiche une comparaison des résultats."""
        print("\n" + "="*70)
        print("📊 RÉSULTATS DU BENCHMARK")
        print("="*70)
        
        ollama = self.results.get("ollama", {})
        tensorrt = self.results.get("tensorrt", {})
        
        if ollama:
            print(f"\n🔵 Ollama ({ollama.get('model', 'N/A')})")
            print(f"  Latence moyenne: {ollama.get('avg_latency', 0):.3f}s")
            print(f"  Débit: {ollama.get('tokens_per_sec', 0):.2f} tokens/s")
        
        if tensorrt:
            print(f"\n🟢 TensorRT ({tensorrt.get('model', 'N/A')})")
            print(f"  Temps de chargement: {tensorrt.get('load_time', 0):.3f}s")
            print(f"  Latence moyenne: {tensorrt.get('avg_latency', 0):.3f}s")
            print(f"  Débit: {tensorrt.get('tokens_per_sec', 0):.2f} tokens/s")
        
        if ollama and tensorrt:
            speedup = ollama["avg_latency"] / tensorrt["avg_latency"] if tensorrt["avg_latency"] > 0 else 0
            throughput_gain = (tensorrt["tokens_per_sec"] - ollama["tokens_per_sec"]) / ollama["tokens_per_sec"] * 100 if ollama["tokens_per_sec"] > 0 else 0
            
            print(f"\n📈 AMÉLIORATION")
            print(f"  Accélération: {speedup:.2f}x")
            print(f"  Gain de débit: {throughput_gain:+.1f}%")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    benchmark = Benchmark()
    
    # Test prompts
    test_prompts = [
        "Can you give me the banana in front of you?",
        "Pick up the red car and avoid the obstacles.",
        "What do you see on the table?"
    ]
    
    print("🚀 Démarrage du benchmark BIRA - Ollama vs TensorRT\n")
    
    # Benchmark Ollama
    try:
        ollama_stats = benchmark.benchmark_ollama(
            model_name="BIRA",
            test_prompts=test_prompts,
            num_runs=3
        )
    except Exception as e:
        logger.error(f"Erreur Ollama benchmark: {e}")
    
    # Benchmark TensorRT
    try:
        tensorrt_stats = benchmark.benchmark_tensorrt(
            test_prompts=test_prompts,
            num_runs=3
        )
    except Exception as e:
        logger.error(f"Erreur TensorRT benchmark: {e}")
    
    # Afficher la comparaison
    benchmark.compare_results()
