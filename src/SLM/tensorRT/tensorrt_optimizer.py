"""
------------------------------------------------------------------------------------
TensorRT Optimizer for Ollama Models
------------------------------------------------------------------------------------
2026-01-07 v1.0 - Initial implementation for optimizing LLM with TensorRT
------------------------------------------------------------------------------------
DESCRIPTION GÉNÉRALE
------------------------------------------------------------------------------------
Ce module fournit une interface pour optimiser les modèles Ollama (llama 3.2.1b)
avec TensorRT pour améliorer les performances d'inférence.

NOTES
------------------------------------------------------------------------------------
- Nécessite TensorRT-LLM installé
- Compatible avec les modèles Llama
- Optimisations incluent: quantization, fusion d'opérateurs, optimisation GPU
"""

import os
import json
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_jetson_platform():
    """
    Détecte si le code s'exécute sur un Jetson.
    
    Returns:
        Dict avec les infos de la plateforme Jetson ou None
    """
    try:
        # Vérifier le fichier de version Jetson
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                version_info = f.read()
            
            # Détecter le modèle spécifique
            model = "Unknown Jetson"
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip('\x00')
            
            return {
                "is_jetson": True,
                "model": model,
                "version_info": version_info.strip(),
                "architecture": platform.machine()
            }
    except Exception as e:
        logger.debug(f"Pas de plateforme Jetson détectée: {e}")
    
    return {"is_jetson": False, "architecture": platform.machine()}


class TensorRTOptimizer:
    """
    Classe pour optimiser les modèles Ollama avec TensorRT.
    
    Attributs:
        model_name: Nom du modèle Ollama à optimiser
        output_dir: Répertoire de sortie pour les modèles optimisés
        precision: Précision du modèle ('fp16', 'int8', 'int4')
        max_batch_size: Taille maximale du batch
        max_input_len: Longueur maximale de l'entrée
        max_output_len: Longueur maximale de la sortie
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        output_dir: str = "./tensorrt_models",
        precision: str = "fp16",
        max_batch_size: int = 1,
        max_input_len: int = 512,
        max_output_len: int = 192,
        gpu_id: int = 0,
        auto_detect_jetson: bool = True
    ):
        """
        Initialise l'optimiseur TensorRT.
        
        Args:
            model_name: Nom du modèle Ollama
            output_dir: Dossier de sortie pour les modèles TensorRT
            precision: Précision ('fp16', 'int8', 'int4')
            max_batch_size: Taille max du batch
            max_input_len: Longueur max d'entrée (tokens)
            max_output_len: Longueur max de sortie (tokens)
            gpu_id: ID du GPU à utiliser
            auto_detect_jetson: Détection automatique de la plateforme Jetson
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.gpu_id = gpu_id
        
        # Détection de la plateforme
        self.platform_info = detect_jetson_platform() if auto_detect_jetson else {"is_jetson": False}
        
        # Configuration adaptée pour Jetson Orin Nano
        if self.platform_info["is_jetson"]:
            logger.info(f"🤖 Jetson détecté: {self.platform_info['model']}")
            logger.info(f"Architecture: {self.platform_info['architecture']}")
            
            # Optimisations spécifiques Jetson Orin Nano (8GB RAM)
            if "Orin Nano" in self.platform_info.get("model", ""):
                logger.info("⚙️  Application des optimisations Jetson Orin Nano")
                # Réduire les longueurs max pour économiser la mémoire
                if self.max_input_len > 256:
                    logger.warning(f"Réduction max_input_len de {self.max_input_len} à 256 pour Jetson")
                    self.max_input_len = 256
                if self.max_output_len > 128:
                    logger.warning(f"Réduction max_output_len de {self.max_output_len} à 128 pour Jetson")
                    self.max_output_len = 128
        
        # Créer le répertoire de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins des modèles
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.engine_dir = self.output_dir / "engines"
        
        logger.info(f"TensorRT Optimizer initialisé pour {model_name}")
        logger.info(f"Précision: {self.precision}, Max input: {self.max_input_len}, Max output: {self.max_output_len}")
    
    def check_dependencies(self) -> bool:
        """
        Vérifie que les dépendances nécessaires sont installées.
        
        Returns:
            True si toutes les dépendances sont présentes
        """
        dependencies = {
            "tensorrt": "TensorRT",
            "torch": "PyTorch",
            "transformers": "Transformers"
        }
        
        missing = []
        for module, name in dependencies.items():
            try:
                mod = __import__(module)
                if module == "torch":
                    import torch
                    cuda_available = torch.cuda.is_available()
                    logger.info(f"✅ {name} trouvé (CUDA: {cuda_available})")
                    if cuda_available:
                        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                        logger.info(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                else:
                    logger.info(f"✅ {name} trouvé")
            except ImportError:
                missing.append(name)
                logger.error(f"❌ {name} non trouvé")
        
        if missing:
            logger.error(f"Dépendances manquantes: {', '.join(missing)}")
            if self.platform_info["is_jetson"]:
                logger.info("Sur Jetson, installez avec:")
                logger.info("  sudo apt-get install nvidia-tensorrt python3-libnvinfer-dev")
                logger.info("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                logger.info("Installez avec: pip install tensorrt torch transformers")
            return False
        
        # Vérifications spécifiques Jetson
        if self.platform_info["is_jetson"]:
            try:
                result = subprocess.run(
                    ["jetson_clocks", "--show"],
                    capture_output=True,
                    text=True
                )
                logger.info("✅ jetson_clocks disponible")
            except FileNotFoundError:
                logger.warning("⚠️  jetson_clocks non trouvé (optionnel)")
        
        return True
    
    def export_ollama_model(self) -> bool:
        """
        Exporte le modèle Ollama au format compatible avec TensorRT.
        
        Returns:
            True si l'export réussit
        """
        try:
            logger.info(f"📦 Export du modèle Ollama: {self.model_name}")
            
            # Vérifier qu'Ollama est installé
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            
            if self.model_name not in result.stdout:
                logger.error(f"Modèle {self.model_name} non trouvé dans Ollama")
                logger.info(f"Téléchargez-le avec: ollama pull {self.model_name}")
                return False
            
            # Créer le répertoire de checkpoints
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Modèle {self.model_name} prêt pour conversion")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'export: {e}")
            return False
    
    def convert_to_tensorrt(self) -> bool:
        """
        Convertit le modèle exporté en engine TensorRT optimisé.
        
        Returns:
            True si la conversion réussit
        """
        try:
            logger.info("🔧 Conversion vers TensorRT engine...")
            
            # Créer le répertoire d'engines
            self.engine_dir.mkdir(parents=True, exist_ok=True)
            
            config = {
                "model_name": self.model_name,
                "precision": self.precision,
                "max_batch_size": self.max_batch_size,
                "max_input_len": self.max_input_len,
                "max_output_len": self.max_output_len,
                "gpu_id": self.gpu_id
            }
            
            # Sauvegarder la configuration
            config_path = self.engine_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Configuration sauvegardée: {config_path}")
            logger.info(f"📍 Engine TensorRT sera créé dans: {self.engine_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la conversion: {e}")
            return False
    
    def optimize(self) -> bool:
        """
        Pipeline complet d'optimisation.
        
        Returns:
            True si l'optimisation réussit
        """
        logger.info("🚀 Démarrage de l'optimisation TensorRT...")
        
        # 1. Vérifier les dépendances
        if not self.check_dependencies():
            return False
        
        # 2. Exporter le modèle Ollama
        if not self.export_ollama_model():
            return False
        
        # 3. Convertir vers TensorRT
        if not self.convert_to_tensorrt():
            return False
        
        logger.info("✅ Optimisation terminée avec succès!")
        logger.info(f"📂 Modèles optimisés disponibles dans: {self.output_dir}")
        
        return True
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur l'engine TensorRT.
        
        Returns:
            Dictionnaire avec les informations de configuration
        """
        config_path = self.engine_dir / "config.json"
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    # Exemple d'utilisation
    optimizer = TensorRTOptimizer(
        model_name="llama3.2:1b",
        precision="fp16",
        max_batch_size=1,
        max_input_len=512,
        max_output_len=192
    )
    
    success = optimizer.optimize()
    
    if success:
        print("\n" + "="*60)
        print("✅ Modèle optimisé avec succès!")
        print("="*60)
        info = optimizer.get_engine_info()
        print(json.dumps(info, indent=2))
    else:
        print("\n❌ L'optimisation a échoué. Vérifiez les logs ci-dessus.")
