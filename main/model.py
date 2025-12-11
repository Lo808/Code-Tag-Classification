import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import AdamW
from tqdm import tqdm
from evaluation import compute_f1_scores, per_tag_f1


class CodeBERTClassifier(nn.Module):
    
    def __init__(self,mlb_classes, model_name='microsoft/codebert-base'):
        super(CodeBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.mlb_classes=mlb_classes
        self.best_thresholds=None

        # Dropout to prevent overfitting, quite high because few training data
        self.drop = nn.Dropout(p=0.3)
        
        # Add a final layer than maps Bert output to classification over the tags
        self.out = nn.Linear(self.bert.config.hidden_size,len(self.mlb_classes))
        
    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        output = self.drop(cls_token)

        return self.out(output)
    
    def fit(self, train_loader, dev_loader, epochs=4, lr=2e-5, device='cuda', save_path='best_model.pt',pos_weight=None):
        """
        Train model and save best version
        """
        # Configuration
        self.to(device)

        # Optimiseur for transformers AdamW with Weight Decay, more robust to overfitting
        optimizer = AdamW(self.parameters(), lr=lr)

        # Use of a weighted loss to deal with imbalanced classes and penalize more errors in less represented classes
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        print(f"Démarrage de l'entraînement sur {device} pour {epochs} époques...")

        for epoch in range(epochs):

            # Put model in training mode: active dropout
            self.train() 
            total_train_loss = 0
            
            #Loading bar
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_loop:
                # Move to Device
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) # Doit être float

                # Forward Pass
                optimizer.zero_grad() # Reset des gradients
                outputs = self(input_ids, mask) # Logits

                loss = criterion(outputs, labels)
                
                # Backward Pass
                loss.backward()
                
                # Gradient Clipping to prevent gradient exploding
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update Weights with gradient descent
                optimizer.step()

                total_train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase for the model

            self.eval() # Put in eval, no dropout
            total_val_loss = 0
            
            with torch.no_grad(): # No gradient stored here
                for batch in dev_loader:
                    input_ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self(input_ids, mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(dev_loader)
            
            #Logs
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save this version if better
            if avg_val_loss < best_val_loss:
                print(f"✅ Amélioration ! (Loss: {best_val_loss:.4f} -> {avg_val_loss:.4f}). Sauvegarde...")
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), save_path)
        
        print("\nEntraînement terminé.")
        # Load best model to predict next
        self.load_state_dict(torch.load(save_path))

        return history
    
    def tune_thresholds(self, dev_loader, device):
        """
        Optimisation Post-Training : Find optimal threshold for each class (0.1-0.9) 

        Returns:
            Dict: {tag: f1 score}
        """
        from sklearn.metrics import f1_score

        self.eval()
        self.to(device)
        
        # Get all proba from the raw predict method
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self(input_ids, mask)
                probs = torch.sigmoid(logits) # Conversion logits -> [0, 1]
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        y_prob = np.vstack(all_probs)
        y_true = np.vstack(all_targets)
        
        # 2. Grid Search per class
        thresholds_range = np.arange(0.1, 0.95, 0.05) # Teste 0.10, 0.15 ... 0.90
        best_thresholds = {}
        
        for i, tag_name in enumerate(self.mlb_classes):
            best_f1 = 0
            best_thresh = 0.5
            
            y_true_col = y_true[:, i]
            y_prob_col = y_prob[:, i]
            
            # Test threshold and keep best one yet
            for t in thresholds_range:
                y_pred_col = (y_prob_col >= t).astype(int)
                f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
            
            best_thresholds[tag_name] = round(best_thresh, 2)
            print(f"   🔹 Tag: {tag_name:<15} | Seuil Optimal: {best_thresh:.2f} | F1: {best_f1:.3f}")
            
        self.best_thresholds = best_thresholds
        return best_thresholds

    def predict(self, data_loader, device):
        """
        Predict tag according to model weight
        """
        
        self.eval()
        self.to(device)
        all_preds = []
        
        # If no optimal threshold back to default value 0.5
        if not hasattr(self, 'best_thresholds') or self.best_thresholds is None:
            print("⚠️ Attention: Seuils non calibrés. Utilisation de 0.5 par défaut.")
            thresholds_tensor = torch.full((len(self.mlb_classes),), 0.5).to(device)
        else:
            # Use best thresholds
            thresholds_list = [self.best_thresholds[tag] for tag in self.mlb_classes]
            thresholds_tensor = torch.tensor(thresholds_list).to(device)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                
                logits = self(input_ids, mask)
                probs = torch.sigmoid(logits)
                
                # Use good threshold to form predictions
                preds = (probs >= thresholds_tensor).float()
                
                all_preds.append(preds.cpu().numpy())
                
        return np.vstack(all_preds)
    
    def evaluate_model(self, data_loader, device, target_tags):
        """
        Give performance indicators for the model
        """
        
        # Get predictions
        y_pred_numpy = self.predict(data_loader, device)
        
        # Get real labels
        all_labels = []
        for batch in data_loader:
            all_labels.append(batch['labels'].cpu().numpy())
        y_true_numpy = np.vstack(all_labels)
        
        # Indicators
        print("\n--- Métriques Globales ---")
        global_metrics = compute_f1_scores(y_true_numpy, y_pred_numpy)
        print(global_metrics)
        
        print("\n--- Métriques par Tag ---")
        tag_results = per_tag_f1(y_true_numpy, y_pred_numpy, self.mlb_classes, focus_tags=target_tags)
        
        for tag, score in tag_results:
            print(f"{tag:<15} : {score}")
            
        return global_metrics, tag_results