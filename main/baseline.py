from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


class tf_idf():
    '''
    Docstring for tf_idf
    Simple inverse frequency model as baseline, use vectorizer and Multilabel Binarizer
    A very important parameter is the threshold value for classifying
    '''
    def __init__(self,max_features=100_000, ngrams=(1, 2),C=1.0):
        self.focus_tags=['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
        self.vectorizer=TfidfVectorizer(
        lowercase=False,        # we already lowercase manually
        ngram_range=ngrams,
        max_features=max_features,
        strip_accents="unicode")

        self.mlb=MultiLabelBinarizer()

        self.model=OneVsRestClassifier(LogisticRegression(max_iter=300, C=C))
        self.is_fitted=False
        self.best_thresholds=None
        
        pass


    def fit(self,text,labels):
        
        X=self.vectorizer.fit_transform(text)
        y=self.mlb.fit_transform(labels)
        self.model.fit(X,y)
        self.is_fitted=True

    def predict_probas(self,test_text):
        """
        Predict probas label matrix for the provided texts.
        Assumes the model has already been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        # Use the existing vocabulary; do not refit on test data
        X_test=self.vectorizer.transform(test_text)
        probas=self.model.predict_proba(X_test)

        return probas
    
    def predict(self,test_text,threshold=0.3):
        """
        Predict binary label matrix for the provided texts.
        Assumes the model has already been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        if getattr(self,'best_thresholds',None) is not None:
            threshold=self.best_thresholds

        
        probas=self.predict_probas(test_text)

        # Case 1: Single threshold
        if not isinstance(threshold, dict):
            return (probas >= threshold).astype(int)

        # Case 2: Using tuned macro or per class threshold
        y_pred=np.zeros_like(probas, dtype=int)
        for c,tag in enumerate(self.mlb.classes_):
            thr = threshold[tag]          # retrieve threshold for this tag
            y_pred[:, c] = (probas[:, c] >= thr).astype(int)

        return y_pred
    
    def tune_threshold(self,dev_text,dev_labels,plot=False,depth=20):
        '''
        Tune hyperparameter threshold in predict method to get best f1score
        
        :param plot: Description
        :param depth: Precision of the interval between p=0.05 and p=0.5
        '''

        f1_scores=[]
        thresholds=np.linspace(0.05,0.50,depth)

        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        # Get prediction for text
        y_true=self.mlb.transform(dev_labels)
        probas=self.predict_probas(dev_text)

        # Compare threshold performances
        for thresh in thresholds:
            y_pred=y_pred=(probas> thresh).astype(int)
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
        best_f1=f1_scores[best_thresh_index]

        n_classes=len(self.mlb.classes_)
        self.best_thresholds=np.full(n_classes, best_thresh)
        return best_thresh,best_f1
    
    def tune_per_tag_threshold(self,dev_text,dev_labels,depth=20):
        '''
        Due to sever imbalance in class representation, we might need to tune 
        the threshold per class to get best f1score.
        '''

        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        # Init logs
        best_thresholds={}
        best_f1_per_class={}
        thresholds=np.linspace(0.05,0.50,depth) 

        # Get prediction for text
        y_true=self.mlb.transform(dev_labels)
        probas=self.predict_probas(dev_text)

        # Go through all class and compute best threshold
        n_classes=y_true.shape[1]
        for c in range(n_classes):
            tag = self.mlb.classes_[c] 
            y_true_c=y_true[:,c]
            probas_c = probas[:, c]
            best_class_f1=0
            best_class_thresh=0.5

            for thresh in thresholds:
                 
                y_pred_c = (probas_c >= thresh).astype(int)
                f1 = f1_score(y_true_c, y_pred_c, average="binary", zero_division=0)
                
                if f1 > best_class_f1:
                    best_class_f1 = f1
                    best_class_thresh = thresh

            
            best_thresholds[tag] = best_class_thresh
            best_f1_per_class[tag] = best_class_f1
        
        self.best_thresholds=best_thresholds

        return best_thresholds, best_f1_per_class





