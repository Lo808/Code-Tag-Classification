import numpy as np
import torch
import torch.autograd as ag
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from transformers import AutoModel


class CodeBERTClassifier(nn.Module):
    def __init__(self, n_classes=8, model_name='microsoft/codebert-base'):
        super(CodeBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Dropout to prevent overfitting, quite high because few training data
        self.drop = nn.Dropout(p=0.3)
        
        # Add a final layer than maps Bert output to classification over the tags
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        # Use of a weighted loss to deal with imbalanced classes and penalize more errors in less represented classes


    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        output = self.drop(cls_token)

        return self.out(output)
    
    def get_pos_weight(self,train_loader):

        all_labels = []
    
        # We iterate through the loader without calculating gradients (faster)
        for batch in train_loader:
            # Assuming your batch is a dictionary like {'input_ids': ..., 'labels': ...}
            labels = batch['labels']
            all_labels.append(labels)
    
        #   Concatenate all batches into one large tensor and convert to numpy
        train_labels_bin=torch.cat(all_labels).cpu().numpy()
        num_positives = np.sum(train_labels_bin, axis=0)
        num_negatives = len(train_labels_bin) - num_positives

        # Sécurité : on évite la division par zéro (si un tag n'apparait jamais)
        # On remplace les 0 par 1 temporairement
        num_positives = np.clip(num_positives, a_min=1, a_max=None)

        # Formule classique pour balancer les classes
        weights = num_negatives / num_positives
        pos_weights_tensor = torch.tensor(weights, dtype=torch.float)

        return pos_weights_tensor

    def fit(self, train_loader, val_loader, epochs=4, lr=2e-5, device='cuda', save_path='best_model.pt',pos_weight=True):
        """
        Train model and save best version
        """



        # Configuration
        self.to(device)

        # Optimiseur for transformers AdamW with Weight Decay, more robust to overfitting
        optimizer = AdamW(self.parameters(), lr=lr)

        # Loss 
        if pos_weight:
            pos_weights_tensor=self.get_pos_weight(train_loader)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
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
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self(input_ids, mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            
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