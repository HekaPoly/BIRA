from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import re

# --------------------------------------------------------------------------------------
#  Classe Formater
# --------------------------------------------------------------------------------------
@dataclass
class Formater:
    """
    Le `Formatter` sert à préparer le prompt envoyé au modèle, à extraire la
    première structure JSON renvoyée, puis à normaliser les données pour produire
    un dictionnaire cohérent avec les clés attendues par la classe `Extraction`.

    Étapes :
      1) build_prompt(user_cmd) : formate une consigne claire pour le SLM.
      2) parse(raw_text) : isole le premier bloc JSON de la sortie texte brute.
      3) postprocess(raw, user_cmd) : normalise et complète les champs manquants.
    """

    def build_prompt(self, user_cmd: str) -> str:
        """
        Construit un prompt explicite demandant au modèle d’extraire uniquement un JSON.

        Parameters
        ----------
        user_cmd : str
            Commande ou phrase en langage naturel saisie par l’utilisateur.

        Returns
        -------
        str
            Prompt à envoyer au modèle Ollama.
        """
        return (
            "Tu es un extracteur d'informations. Retourne UNIQUEMENT un JSON compact "
            'avec les clés: response (str), target_object (str ou null), obstacles (liste de str), '
            'status ("ok" | "missing_target" | "ambiguous" | "empty"), confidence (0..1). '
            "Règles STRICTES: "
            "1) 'response' = courte phrase polie à la 1re personne qui confirme l’action (ex: « Je te donne la pomme bleue. »). "
            "2) NE JAMAIS répéter la phrase de l’utilisateur ni commencer par « BIRA, ... ». "
            "3) 'target_object' = le nom de l’objet demandé (ex: « pomme », « tasse », « banane »). Pas de pronom, pas de lieu, pas seulement une couleur. Si plusieurs noms sont présents, choisis l’objet associé au verbe (donner/attraper/apporter/passer). Si une couleur est donnée avec l’objet, inclue-la (ex: « pomme verte »). "
            "4) 'obstacles' = autres noms cités qui ne sont PAS la cible (repères, supports), ex: « à côté de l’ordinateur » => obstacle = « ordinateur ». "
            "5) Si l’utilisateur donne seulement un nom d’objet (ex: « la pomme »), accepte-le comme target_object (ex: \"pomme\") au lieu de null. "
            "6) JSON valide et auto-contenu, aucun texte autour. "
            "7) Lis les exemples mais RÉPONDS UNIQUEMENT pour la ligne TEXTE_CIBLE et ne copie pas mot pour mot.\n"
            # --- Exemples variés ---
            'Texte: "Attrape la voiture rouge et évite le bus à gauche"\n'
            '{"response":"Je prends la voiture rouge.","target_object":"voiture rouge","obstacles":["bus"],"status":"ok","confidence":0.86}\n'
            'Texte: "Apporte-moi la balle jaune"\n'
            '{"response":"Je t’apporte la balle jaune.","target_object":"balle jaune","obstacles":[],"status":"ok","confidence":0.85}\n'
            'Texte: "Donne la pomme bleue devant l’ordinateur"\n'
            '{"response":"Je te donne la pomme bleue.","target_object":"pomme bleue","obstacles":["ordinateur"],"status":"ok","confidence":0.9}\n'
            'Texte: "Donne-moi la pomme"\n'
            '{"response":"Je te donne la pomme.","target_object":"pomme","obstacles":[],"status":"ok","confidence":1.0}\n'
            'Texte: "Je veux la pomme à côté de l’ordinateur"\n'
            '{"response":"Je te donne la pomme à côté de l’ordinateur.","target_object":"pomme","obstacles":["ordinateur"],"status":"ok","confidence":0.9}\n'
            'Texte: "Passe-moi la tasse sur la table"\n'
            '{"response":"Je te passe la tasse.","target_object":"tasse","obstacles":["table"],"status":"ok","confidence":0.88}\n'
            # --- Cible ---
            f"TEXTE_CIBLE: {user_cmd}\n"
            "JSON:"
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Extrait le premier objet JSON valide contenu dans la sortie du modèle.

        Parameters
        ----------
        raw_text : str
            Texte brut renvoyé par le modèle.

        Returns
        -------
        dict
            Dictionnaire Python décodé à partir du JSON extrait.
        """
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Aucun JSON détecté dans la sortie du modèle.")
        return json.loads(raw_text[start:end + 1])

    def postprocess(self, raw: Dict[str, Any], user_cmd: str) -> Dict[str, Any]:
        """
        Normalise les champs du dictionnaire brut pour assurer la cohérence du schéma.

        Returns
        -------
        dict
            Dictionnaire contenant les champs normalisés : response, target_object,
            obstacles, status et confidence.
        """

        # Cas explicite : commandes hors extraction (ex. "stop")
        if user_cmd.strip().lower() in {"stop", "arrete", "arrête"}:
            return {
                "response": "Je reste en attente.",
                "target_object": None,
                "obstacles": [],
                "status": "empty",
                "confidence": 0.9,
            }

        # Petits échanges sociaux (merci, salut) : réponse courte et pas de cible
        small_talk = {"merci", "thanks", "salut", "hello", "allo", "bonjour", "bonsoir", "ca va", "ça va"}
        if not raw.get("target_object") and any(token in user_cmd.lower() for token in small_talk):
            return {
                "response": "Avec plaisir !",
                "target_object": None,
                "obstacles": [],
                "status": "empty",
                "confidence": 0.9,
            }

        REFORMULATION_MSG = (
            "Je ne suis pas sûr de bien comprendre (confiance ≈{pct}%). "
            "Peux-tu reformuler en précisant l’objet cible (nom + couleur), "
            "sa position (ex.: « devant l’ordinateur, derrière toi ») et, s’il y en a, "
            "les obstacles à éviter ?"
        )

        THRESHOLD = 0.60

        response = (raw.get("response") or "").strip()
        target_object = raw.get("target_object", None)
        obstacles = raw.get("obstacles") or []
        status = (raw.get("status") or "ok").strip()
        # Model-provided confidence can be noisy; derive a simple heuristic score and blend
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Normaliser le target_object brut
        target_object = self._normalize_target(target_object)

        # If the model returned a stopword/pronominal/landmark as target, try to recover
        STOP_TARGETS = {
            "bira", "toi", "moi", "va",
            "ordinateur", "lordinateur", "ordi",
            "telephone", "téléphone", "tel",
            "table", "cote", "côté",
        }
        if isinstance(target_object, str) and target_object.strip().lower() in STOP_TARGETS:
            recovered = self._guess_target_from_user_cmd(user_cmd)
            if recovered:
                target_object = recovered
                if status in {"missing_target", "ambiguous", "empty"}:
                    status = "ok"
                if not response:
                    response = f"Je te donne {target_object}."

        # If the model returned only a color as target, try to replace with a noun phrase from the user command
        COLOR_ONLY = {
            "bleu", "bleue", "rouge", "jaune", "vert", "verte",
            "noir", "noire", "blanc", "blanche", "gris", "grise",
            "violet", "violette", "rose", "orange",
        }
        if isinstance(target_object, str) and target_object.strip().lower() in COLOR_ONLY:
            guessed_color_target = self._guess_target_from_user_cmd(user_cmd)
            if guessed_color_target:
                target_object = guessed_color_target
                if status in {"missing_target", "ambiguous", "empty"}:
                    status = "ok"
                if not response:
                    response = f"Je te donne {target_object}."

        # Try to infer a target when the model did not return one
        guessed_target: Optional[str] = None
        if not target_object:
            guessed_target = self._guess_target_from_user_cmd(user_cmd)
            if guessed_target:
                target_object = guessed_target
                if status in {"missing_target", "ambiguous", "empty"}:
                    status = "ok"
                if not response:
                    response = f"Je te donne {target_object}."

        # Si le modèle a renvoyé un message de reformulation alors que la cible est identifiée, remplacer par une réponse affirmative
        if target_object and response.lower().startswith("peux-tu reformuler"):
            response = f"Je te donne {target_object}."
            if status in {"ambiguous", "missing_target"}:
                status = "ok"

        # Si la réponse ne contient pas la cible normalisée, régénérer une réponse propre
        if target_object and target_object not in response.lower():
            response = f"Je te donne {target_object}."

        # Heuristic confidence: reward presence of target_object and valid obstacles
        heuristic = 0.1  # base
        if target_object:
            heuristic += 0.5
        if obstacles:
            heuristic += 0.1
        if status == "ok":
            heuristic += 0.2
        heuristic = max(0.0, min(1.0, heuristic))

        # Blend to reduce hallucinated confidence spikes (weight heuristic higher)
        confidence = round(0.7 * heuristic + 0.3 * confidence, 2)

        # Forcer obstacles à être une liste de strings propres
        if not isinstance(obstacles, list):
            obstacles = []
        obstacles = [str(o).strip() for o in obstacles if str(o).strip()]

        # 2) Appliquer la règle de reformulation
        if (confidence < THRESHOLD) or (status in {"missing_target", "ambiguous", "empty"}):
            # Si le modèle disait "ok" mais confiance faible, on bascule en "ambiguous"
            if status == "ok":
                status = "ambiguous"
            pct = int(round(confidence * 100))
            response = REFORMULATION_MSG.format(pct=pct)

        # 3) Valeur par défaut pour response si vide et status ok (rare)
        if not response and status == "ok":
            response = "D’accord."

        return {
            "response": response,
            "target_object": target_object,
            "obstacles": obstacles,
            "status": status,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    def _guess_target_from_user_cmd(self, user_cmd: str) -> Optional[str]:
        """
        Heuristic extraction of a target noun phrase when the model omits it.
        """
        if not user_cmd:
            return None

        text = user_cmd.lower()
        text = re.sub(r"[.,;!?]", " ", text)

        patterns = [
            r"(?:donne[- ]?moi|apporte[- ]?moi|passe[- ]?moi|donne)\s+(?:la|le|les|un|une|des)?\s*(.+)",
            r"(?:veux[- ]?tu|peux[- ]?tu)\s+(?:me\s+)?(?:donner|apporter|passer)\s+(?:la|le|les|un|une|des)?\s*(.+)",
            r"(?:je\s+veux|je\s+voudrais)\s+(?:la|le|les|un|une|des)?\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                candidate = m.group(1).strip()
                candidate = re.split(
                    r"\s+(?:s'il|stp|svp|merci|derriere|derrière|devant|sur|dans|a\s+cote|à\s+côté|cote|côté|pres|près|qui|que|dont)\b",
                    candidate,
                )[0].strip()
                if candidate:
                    return candidate

        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9'-]+", text)
        stop = {
            "bira", "donne", "donne-moi", "moi", "une", "un", "la", "le", "les", "des",
            "stp", "svp", "merci", "s", "il", "te", "plait", "plaît",
            "toi", "va", "veux", "voudrais",
            "ordinateur", "lordinateur", "ordi", "telephone", "téléphone", "tel",
            "cote", "côté",
        }
        tokens = [t for t in tokens if t not in stop]
        return tokens[-1] if tokens else None

    # ------------------------------------------------------------------
    def _normalize_target(self, target: Any) -> Optional[str]:
        """Nettoie le target_object: enlève articles/pronoms/residus."""
        if not target:
            return None
        t = str(target).strip()
        t = re.sub(r"[\"'`]", "", t)
        t = re.sub(r"\s+", " ", t)
        t = t.lower()
        t = re.sub(r"^(?:l|d)[aeou]s?\s+", "", t)  # l', d', loosely
        t = re.sub(r"^(?:une|un|la|le|les|des)\s+", "", t)
        t = re.sub(r"^(?:e\s+)", "", t)  # cas "e pomme"
        t = t.strip(". ")
        return t or None
