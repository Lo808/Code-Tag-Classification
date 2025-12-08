from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import joblib
import numpy as np


class tf_idf():
    '''
    Docstring for tf_idf
    Simple inverse frequency model as baseline, use vectorizer and Multilabel Binarizer
    A very important parameter is the threshold value for classifying
    '''
    def __init__(self,max_features=100_000, ngrams=(1, 2),C=1.0):
        
        self.vectorizer=TfidfVectorizer(
        lowercase=False,        # we already lowercase manually
        ngram_range=ngrams,
        max_features=max_features,
        strip_accents="unicode")

        self.mlb=MultiLabelBinarizer()

        self.model=OneVsRestClassifier(LogisticRegression(max_iter=300, C=C))
        self.is_fitted=False
        pass


    def fit(self,text,labels):
        
        X=self.vectorizer.fit_transform(text)
        y=self.mlb.fit_transform(labels)
        self.model.fit(X,y)
        self.is_fitted=True

    def predict(self,test_text,threshold=0.3):
        """
        Predict binary label matrix for the provided texts.
        Assumes the model has already been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        # Use the existing vocabulary; do not refit on test data
        X_test=self.vectorizer.transform(test_text)
        probas=self.model.predict_proba(X_test)
        y_pred=(probas> threshold).astype(int)

        return y_pred
    
    def save(self, path):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "mlb": self.mlb,
            "model": self.model
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.mlb = data["mlb"]
        self.model = data["model"]
        self.is_fitted = True

    
    def tune_threshold(self,dev_text,dev_labels,plot=False,depth=20):
        f1_scores=[]
        thresholds=np.linspace(0.05,0.50,depth)

        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        y_true=self.mlb.transform(dev_labels)
        
        for thresh in thresholds:

            y_pred=self.predict(dev_text,threshold=thresh)
            f1=f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)
        
        if plot:
            plt.plot(thresholds,f1_scores)
            plt.xlabel("Threshold")
            plt.ylabel("Macro-F1")
            plt.title("Threshold tuning curve")
            plt.grid()
        
        best_thresh_index=np.argmax(f1_scores)
        best_thresh=thresholds[best_thresh_index]
        best_f1=max(f1_scores)

        return best_thresh,best_f1
    
    def tune_per_tag_threshold(self,dev_text,dev_labels,plot=False,depth=20):
        
        f1_scores=[]
        thresholds=np.linspace(0.05,0.50,depth)

        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        y_true=self.mlb.transform(dev_labels)
        
        for thresh in thresholds:

            y_pred=self.predict(dev_text,threshold=thresh)
            f1=f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)
        
        if plot:
            plt.plot(thresholds,f1_scores)
            plt.xlabel("Threshold")
            plt.ylabel("Macro-F1")
            plt.title("Threshold tuning curve")
            plt.grid()
        
        best_thresh_index=np.argmax(f1_scores)
        best_thresh=thresholds[best_thresh_index]
        best_f1=max(f1_scores)

        return best_thresh,best_f1





