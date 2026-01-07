"""
------------------------------------------------------------------------------------
Jetson Utilities - Outils spécifiques pour Jetson Orin Nano
------------------------------------------------------------------------------------
2026-01-07 v1.0 - Utilitaires pour monitorer et optimiser sur Jetson
------------------------------------------------------------------------------------
DESCRIPTION
------------------------------------------------------------------------------------
Fournit des outils pour:
- Surveiller les performances GPU/CPU
- Configurer les modes de puissance
- Optimiser l'utilisation mémoire
- Vérifier la santé système
"""

import subprocess
import logging
import time
import os
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JetsonMonitor:
    """Classe pour surveiller et optimiser le Jetson Orin Nano."""
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        if not self.is_jetson:
            logger.warning("⚠️  Pas de plateforme Jetson détectée")
    
    def _detect_jetson(self) -> bool:
        """Détecte si on est sur Jetson."""
        return os.path.exists('/etc/nv_tegra_release')
    
    def get_system_info(self) -> Dict:
        """Récupère les informations système du Jetson."""
        if not self.is_jetson:
            return {}
        
        info = {}
        
        try:
            # Version JetPack
            with open('/etc/nv_tegra_release', 'r') as f:
                info['jetpack_version'] = f.read().strip()
        except:
            pass
        
        try:
            # Modèle Jetson
            with open('/proc/device-tree/model', 'r') as f:
                info['model'] = f.read().strip('\x00')
        except:
            pass
        
        try:
            # Mode de puissance
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-q'],
                capture_output=True,
                text=True
            )
            info['power_mode'] = result.stdout
        except:
            pass
        
        return info
    
    def get_gpu_stats(self) -> Dict:
        """Récupère les statistiques GPU en temps réel."""
        if not self.is_jetson:
            return {}
        
        try:
            result = subprocess.run(
                ['tegrastats', '--interval', '1000', '--stop'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            stats_line = result.stdout.strip()
            
            # Parser la ligne tegrastats
            # Format: RAM xxx/xxx MB (lfb xxx) ...
            stats = {'raw': stats_line}
            
            # Extraire RAM
            if 'RAM' in stats_line:
                ram_part = stats_line.split('RAM')[1].split('(')[0].strip()
                stats['ram'] = ram_part
            
            # Extraire GPU usage (GR3D)
            if 'GR3D_FREQ' in stats_line:
                gpu_part = stats_line.split('GR3D_FREQ')[1].split('%')[0].strip()
                stats['gpu_usage'] = gpu_part
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lecture GPU stats: {e}")
            return {}
    
    def set_max_performance(self) -> bool:
        """Active le mode performance maximum."""
        if not self.is_jetson:
            logger.warning("Pas sur Jetson, impossible de changer le mode")
            return False
        
        try:
            # Mode MAXN (15W)
            subprocess.run(
                ['sudo', 'nvpmodel', '-m', '0'],
                check=True,
                capture_output=True
            )
            logger.info("✅ Mode MAXN activé (15W)")
            
            # Verrouiller les fréquences max
            subprocess.run(
                ['sudo', 'jetson_clocks'],
                check=True,
                capture_output=True
            )
            logger.info("✅ Fréquences verrouillées au maximum")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur configuration performance: {e}")
            return False
    
    def check_temperature(self) -> Dict[str, float]:
        """Vérifie les températures du système."""
        temps = {}
        
        if not self.is_jetson:
            return temps
        
        try:
            # Lire les zones thermiques
            thermal_zones = [
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone1/temp',
            ]
            
            for i, zone in enumerate(thermal_zones):
                if os.path.exists(zone):
                    with open(zone, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        temps[f'zone_{i}'] = temp_millidegrees / 1000.0
            
        except Exception as e:
            logger.error(f"Erreur lecture température: {e}")
        
        return temps
    
    def get_memory_usage(self) -> Dict:
        """Récupère l'utilisation mémoire."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem = {}
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    mem['total_kb'] = int(line.split()[1])
                elif 'MemAvailable' in line:
                    mem['available_kb'] = int(line.split()[1])
                elif 'SwapTotal' in line:
                    mem['swap_total_kb'] = int(line.split()[1])
                elif 'SwapFree' in line:
                    mem['swap_free_kb'] = int(line.split()[1])
            
            # Convertir en GB
            mem['total_gb'] = mem.get('total_kb', 0) / (1024 * 1024)
            mem['available_gb'] = mem.get('available_kb', 0) / (1024 * 1024)
            mem['used_gb'] = mem['total_gb'] - mem['available_gb']
            mem['usage_percent'] = (mem['used_gb'] / mem['total_gb']) * 100 if mem['total_gb'] > 0 else 0
            
            return mem
            
        except Exception as e:
            logger.error(f"Erreur lecture mémoire: {e}")
            return {}
    
    def print_status(self):
        """Affiche un rapport complet du système."""
        print("\n" + "="*70)
        print("🤖 JETSON ORIN NANO - STATUS")
        print("="*70)
        
        # Info système
        info = self.get_system_info()
        if info:
            print(f"\n📱 Modèle: {info.get('model', 'N/A')}")
            print(f"📦 JetPack: {info.get('jetpack_version', 'N/A')}")
        
        # Température
        temps = self.check_temperature()
        if temps:
            print(f"\n🌡️  Températures:")
            for zone, temp in temps.items():
                status = "🟢" if temp < 70 else "🟡" if temp < 80 else "🔴"
                print(f"   {status} {zone}: {temp:.1f}°C")
        
        # Mémoire
        mem = self.get_memory_usage()
        if mem:
            print(f"\n💾 Mémoire:")
            print(f"   Total: {mem.get('total_gb', 0):.2f} GB")
            print(f"   Utilisée: {mem.get('used_gb', 0):.2f} GB")
            print(f"   Disponible: {mem.get('available_gb', 0):.2f} GB")
            print(f"   Utilisation: {mem.get('usage_percent', 0):.1f}%")
            
            if mem.get('swap_total_kb', 0) > 0:
                swap_used_gb = (mem['swap_total_kb'] - mem['swap_free_kb']) / (1024 * 1024)
                swap_total_gb = mem['swap_total_kb'] / (1024 * 1024)
                print(f"   Swap: {swap_used_gb:.2f} / {swap_total_gb:.2f} GB")
        
        # GPU Stats
        gpu = self.get_gpu_stats()
        if gpu:
            print(f"\n🎮 GPU:")
            print(f"   RAM: {gpu.get('ram', 'N/A')}")
            if 'gpu_usage' in gpu:
                print(f"   Utilisation: {gpu['gpu_usage']}%")
        
        print("="*70 + "\n")
    
    def optimize_for_inference(self):
        """Configure le système pour l'inférence optimale."""
        print("\n⚙️  Optimisation du système pour l'inférence TensorRT...\n")
        
        # 1. Mode performance max
        if self.set_max_performance():
            logger.info("✅ Mode performance activé")
        
        # 2. Vérifier la température
        temps = self.check_temperature()
        if temps:
            max_temp = max(temps.values())
            if max_temp > 75:
                logger.warning(f"⚠️  Température élevée: {max_temp:.1f}°C")
                logger.warning("   Attendez le refroidissement pour de meilleures performances")
        
        # 3. Vérifier la mémoire
        mem = self.get_memory_usage()
        if mem.get('usage_percent', 0) > 80:
            logger.warning("⚠️  Utilisation mémoire élevée")
            logger.info("   Fermez les applications non nécessaires")
        
        # 4. Afficher le status
        time.sleep(1)
        self.print_status()


def monitor_continuous(interval: int = 2):
    """Surveillance continue des performances."""
    monitor = JetsonMonitor()
    
    if not monitor.is_jetson:
        logger.error("Doit être exécuté sur Jetson")
        return
    
    print("\n🔍 Surveillance continue (Ctrl+C pour arrêter)\n")
    
    try:
        while True:
            monitor.print_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✅ Surveillance arrêtée")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Utilitaires Jetson Orin Nano")
    parser.add_argument('--status', action='store_true', help='Afficher le status')
    parser.add_argument('--optimize', action='store_true', help='Optimiser pour inférence')
    parser.add_argument('--monitor', action='store_true', help='Surveillance continue')
    parser.add_argument('--max-perf', action='store_true', help='Activer performance max')
    
    args = parser.parse_args()
    
    monitor = JetsonMonitor()
    
    if args.status or (not any(vars(args).values())):
        monitor.print_status()
    
    if args.optimize:
        monitor.optimize_for_inference()
    
    if args.max_perf:
        monitor.set_max_performance()
        monitor.print_status()
    
    if args.monitor:
        monitor_continuous()
