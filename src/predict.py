import torch
import json
from transformers import AutoTokenizer
from model import CodeBERTClassifier


class CodeBERTPredictor():
    def __init__(self,mlb_classes,model_path,model_name = "microsoft/codebert-base",threshold_path="best_thresholds.json") -> None:
        self.mlb_classes=mlb_classes
        self.model_path=model_path
        self.threshold_path=threshold_path
        self.model_name=model_name
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.load_prediction_model()
        pass

    def load_prediction_model(self):
        print(f"Chargement du modèle sur {self.device}...")
        
        # get the architecture
        model = CodeBERTClassifier(self.mlb_classes, model_name=self.model_name)
        
        # Load the weights and threshold if possible
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        try:
            with open(self.threshold_path, 'r') as f:
                model.best_thresholds = json.load(f)
            print("✅ Seuils optimisés chargés.")
        except FileNotFoundError:
            print("⚠️ Pas de fichier de seuils trouvé. Utilisation de 0.5 par défaut.")
            model.best_thresholds = None
            
        model.to(self.device)
        model.eval() 
        self.model=model
        


    def predict_single_text(self,text):
        """
        Prend un texte brut (str) et retourne les tags prédits.
        """
        # 1. Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt' 
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        mask = encoding['attention_mask'].to(self.device)
        
        # Prediction
        with torch.no_grad():
            # Get logits from model
            logits=self.model(input_ids, mask)
            probs = torch.sigmoid(logits)
            
            # Apply best thresholds
            if self.model.best_thresholds:
                t_list = [self.model.best_thresholds[tag] for tag in self.mlb_classes]
                thresholds = torch.tensor(t_list).to(self.device)
            else:
                thresholds = torch.tensor([0.5] * len(self.mlb_classes)).to(self.device)
                
            # Get prediction according to correct threshold
            preds_binary = (probs >= thresholds).int().cpu().numpy().flatten()
        
        # Get tags from binary
        predicted_tags = [self.mlb_classes[i] for i, val in enumerate(preds_binary) if val == 1]
        
        return predicted_tags, probs.cpu().numpy().flatten()
    