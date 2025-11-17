from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# --------------------------------------------------------------------------------------
#  Classe Extraction
# --------------------------------------------------------------------------------------
@dataclass
class Extraction:
    """
    Représente la sortie structurée d'une analyse de commande.

    Attributs
    ---------
    response : str
        Message vocal ou textuel à retourner à l’utilisateur.
    target_object : Optional[str]
        Nom exact de l’objet cible (ou None si non détecté).
    obstacles : List[str]
        Liste des obstacles à éviter.
    status : str
        Statut global de l’analyse : "ok", "missing_target", "ambiguous", "empty".
    confidence : float
        Niveau de confiance entre 0 et 1.
    """

    response: str
    target_object: Optional[str]
    obstacles: List[str]
    status: str
    confidence: float = 0.5

    def to_payload(self) -> Dict[str, Any]:
        """
        Retourne une version simplifiée du résultat pour un usage en aval (API, robot…)
        """
        return {
            "response": self.response,
            "target_object": self.target_object,
            "obstacles": self.obstacles,
        }
