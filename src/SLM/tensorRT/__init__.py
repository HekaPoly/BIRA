"""
TensorRT Optimization Module for BIRA SLM
==========================================

Ce module fournit des outils pour optimiser les modèles Ollama avec TensorRT.

Modules disponibles:
- tensorrt_optimizer: Optimisation et conversion des modèles
- tensorrt_inference: Moteur d'inférence TensorRT
- benchmark_tensorrt: Benchmarking des performances

Exemple d'utilisation:
    from SLM.tensorRT import TensorRTOptimizer, TensorRTInferenceEngine
    
    # Optimiser le modèle
    optimizer = TensorRTOptimizer(model_name="llama3.2:1b")
    optimizer.optimize()
    
    # Utiliser le moteur d'inférence
    engine = TensorRTInferenceEngine()
    engine.load_engine()
    response = engine.chat(messages=[{"role": "user", "content": "Hello!"}])
"""

from .tensorrt_optimizer import TensorRTOptimizer
from .tensorrt_inference import TensorRTInferenceEngine

__all__ = ["TensorRTOptimizer", "TensorRTInferenceEngine"]
__version__ = "1.0.0"
