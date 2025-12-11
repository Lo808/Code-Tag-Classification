import numpy as np
import torch as th
import torch.autograd as ag
import torch.nn.functional as F


import torch.nn as nn
from transformers import AutoModel

class CodeBERTClassifier(nn.Module):
    def __init__(self, n_classes=8, model_name='microsoft/codebert-base'):

        super(CodeBERTClassifier, self).__init__()
        
        # 1. Chargement du Backbone (le cerveau)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. Dropout pour éviter l'overfitting (très important sur petit dataset)
        self.drop = nn.Dropout(p=0.3)
        
        # Add a final layer to convert into our class number
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Passage dans BERT
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # On récupère le "pooled_output" (le vecteur qui représente toute la phrase)
        # Note : Pour certains modèles comme Roberta/CodeBERT, il vaut mieux prendre output[1] ou le [CLS] token manuellement
        # Ici output.pooler_output fonctionne généralement, sinon output.last_hidden_state[:, 0, :]
        pooled_output = output.pooler_output 
        
        output = self.drop(pooled_output)
        
        # Projection vers les 8 classes
        return self.out(output)
    