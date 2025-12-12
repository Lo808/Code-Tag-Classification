import argparse
import sys
import os
import torch

from predict import CodeBERTPredictor

MODEL_URL = "https://github.com/Lo808/Code-Tag-Classification/releases/download/v1/best_model_with_code_focus.pt"
THRESHOLDS_URL = "https://github.com/Lo808/Code-Tag-Classification/releases/download/v1/best_thresholds_with_code_focus.json"

def check_and_download(path, url):
    if not os.path.exists(path):
        print(f"⬇️ Le fichier '{path}' est manquant. Téléchargement en cours...")
        # Crée le dossier 'models' si besoin
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Télécharge (barre de progression incluse via torch)
        torch.hub.download_url_to_file(url, path)
        print("✅ Téléchargement terminé.")

FOCUS_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']

def main():
    # --- PARTIE 1 : ARGUMENTS (Fusionnée) ---
    parser = argparse.ArgumentParser(
        description="Outil de prédiction CodeBERT (Inférence uniquement)"
    )
    
    parser.add_argument(
        '--text', 
        type=str, 
        required=True, 
        help="Le texte (Description + Code) à classifier entre guillemets."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='models/best_model_with_code_focus.pt', 
        help="Chemin vers le fichier .pt du modèle"
    )
    
    parser.add_argument(
        '--thresholds_path', 
        type=str, 
        default='models/best_thresholds_focus.json', 
        help="Chemin vers le fichier .json des seuils"
    )

    args = parser.parse_args()

    # --- PARTIE 2 : TELECHARGEMENT (Fusionnée) ---
    # Vérification automatique avant de charger
    check_and_download(args.model_path, MODEL_URL)
    check_and_download(args.thresholds_path, THRESHOLDS_URL)

    if not os.path.exists(args.model_path):
        print(f"Le fichier modèle '{args.model_path}' est introuvable.")
        return

    # --- PARTIE 3 : PREDICTION ---
    try:
        predictor = CodeBERTPredictor(
            mlb_classes=FOCUS_TAGS,
            model_path=args.model_path,
            threshold_path=args.thresholds_path
        )
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # J'ai gardé 'predict_single_text' comme dans ton code, 
    # mais vérifie si tu ne l'as pas renommée 'predict' dans ta classe src/predict.py
    tags, scores = predictor.predict_single_text(args.text)

    print("\n" + "="*40)
    print("Résultat de la prédiction")
    print("="*40)
    print(f"📝 Input (50 premiers cars) : {args.text[:50]}...")
    
    if len(tags) > 0:
        print(f"Tags détectés : {tags}")
    else:
        print("Aucun tag détecté (scores trop faibles).")
    
    print("-" * 40)
    print("Détails des scores (Probabilités) :")
    for tag, score in zip(FOCUS_TAGS, scores):
        mark = "🔹" if tag in tags else "  "
        print(f"{mark} {tag:<15} : {score:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()