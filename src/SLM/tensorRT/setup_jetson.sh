#!/bin/bash
###############################################################################
# Script d'installation TensorRT pour Jetson Orin Nano
###############################################################################
# Ce script configure l'environnement TensorRT sur Jetson Orin Nano
# Exécutez avec: bash setup_jetson.sh
###############################################################################

set -e  # Arrêter en cas d'erreur

echo "════════════════════════════════════════════════════════════════════════════"
echo "  Configuration TensorRT pour BIRA sur Jetson Orin Nano"
echo "════════════════════════════════════════════════════════════════════════════"

# Couleurs pour le terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Vérifier qu'on est sur Jetson
echo ""
echo "🔍 Vérification de la plateforme..."
if [ ! -f /etc/nv_tegra_release ]; then
    log_error "Ce script est conçu pour Jetson uniquement!"
    exit 1
fi

JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Jetson Unknown")
log_info "Plateforme détectée: $JETSON_MODEL"

# 2. Vérifier JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(cat /etc/nv_tegra_release)
    log_info "JetPack: $JETPACK_VERSION"
fi

# 3. Mise à jour du système
echo ""
echo "📦 Mise à jour du système..."
sudo apt-get update
log_info "Système mis à jour"

# 4. Installer TensorRT (si nécessaire)
echo ""
echo "🔧 Vérification de TensorRT..."
if dpkg -l | grep -q nvidia-tensorrt; then
    log_info "TensorRT déjà installé"
    dpkg -l | grep TensorRT | head -3
else
    log_warn "Installation de TensorRT..."
    sudo apt-get install -y nvidia-tensorrt python3-libnvinfer-dev
    log_info "TensorRT installé"
fi

# 5. Installer les dépendances Python
echo ""
echo "🐍 Installation des dépendances Python..."

# Vérifier si pip3 est installé
if ! command -v pip3 &> /dev/null; then
    sudo apt-get install -y python3-pip
fi

# Installer les dépendances de base
pip3 install --upgrade pip

# Installer PyTorch (vérifier si wheels locaux existent)
echo ""
echo "🔥 Installation de PyTorch..."
WHEELS_DIR="$HOME/BIRA/wheels"

if [ -f "$WHEELS_DIR/torch-2.3.0-cp310-cp310-linux_aarch64.whl" ]; then
    log_info "Utilisation des wheels locaux PyTorch"
    pip3 install "$WHEELS_DIR/torch-2.3.0-cp310-cp310-linux_aarch64.whl"
    pip3 install "$WHEELS_DIR/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
    pip3 install "$WHEELS_DIR/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl"
else
    log_warn "Wheels locaux non trouvés, installation depuis PyTorch.org..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Installer les autres dépendances
pip3 install transformers accelerate pydantic python-dotenv psutil

# 6. Installer Ollama
echo ""
echo "🦙 Vérification d'Ollama..."
if ! command -v ollama &> /dev/null; then
    log_warn "Installation d'Ollama..."
    curl https://ollama.ai/install.sh | sh
    log_info "Ollama installé"
else
    log_info "Ollama déjà installé"
    ollama --version
fi

# 7. Configuration de la mémoire swap
echo ""
echo "💾 Configuration du swap (recommandé pour LLM)..."
SWAP_SIZE=4G
SWAPFILE=/swapfile

if [ -f "$SWAPFILE" ]; then
    log_info "Swap déjà configuré"
    swapon --show
else
    read -p "Voulez-vous créer un fichier swap de $SWAP_SIZE? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo fallocate -l $SWAP_SIZE $SWAPFILE
        sudo chmod 600 $SWAPFILE
        sudo mkswap $SWAPFILE
        sudo swapon $SWAPFILE
        
        # Rendre permanent
        if ! grep -q "$SWAPFILE" /etc/fstab; then
            echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab
        fi
        
        log_info "Swap créé et activé"
        swapon --show
    fi
fi

# 8. Configuration des performances
echo ""
echo "⚡ Configuration des performances..."
read -p "Activer le mode performance maximum (MAXN 15W)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo nvpmodel -m 0           # Mode MAXN
    sudo jetson_clocks           # Verrouiller les fréquences
    log_info "Mode performance activé"
    
    # Afficher les informations
    sudo nvpmodel -q
fi

# 9. Télécharger le modèle Ollama
echo ""
echo "🤖 Téléchargement du modèle Ollama..."
read -p "Télécharger llama3.2:1b? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ollama pull llama3.2:1b
    log_info "Modèle téléchargé"
fi

# 10. Vérification finale
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  Vérification de l'installation"
echo "════════════════════════════════════════════════════════════════════════════"

# Python
echo ""
python3 --version

# PyTorch + CUDA
echo ""
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || log_error "PyTorch non installé correctement"

# TensorRT
echo ""
python3 -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')" 2>/dev/null || log_error "TensorRT non accessible depuis Python"

# Ollama
echo ""
ollama list

# Stats système
echo ""
echo "📊 Statistiques système:"
free -h
echo ""
tegrastats --interval 1000 --logfile /dev/stdout --stop 2>&1 | head -1 || log_warn "tegrastats non disponible"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
log_info "Installation terminée!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Prochaines étapes:"
echo "  1. cd ~/BIRA/src"
echo "  2. python3 -m SLM.tensorRT.tensorrt_optimizer"
echo "  3. python3 -m SLM.tensorRT.benchmark_tensorrt"
echo ""
echo "Pour surveiller les performances:"
echo "  tegrastats       # Statistiques en temps réel"
echo "  jtop             # Interface graphique (installez avec: pip3 install jetson-stats)"
echo ""
