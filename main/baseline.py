from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib



class tf_idf():
    '''
    Docstring for tf_idf
    Simple inverse frequency model as baseline, use vectorizer and Multilabel Binarizer

    '''
    def __init__(self,max_features=40000, ngrams=(1, 2),C=1.0):
        
        self.vectorizer=TfidfVectorizer(
        lowercase=False,        # we already lowercase manually
        ngram_range=ngrams,
        max_features=max_features,
        strip_accents="unicode"
    )
        self.mlb=MultiLabelBinarizer()

        self.model = OneVsRestClassifier(LogisticRegression(max_iter=300, C=C))
        self.is_fitted=False
        pass


    def fit(self,text,labels):
        
        X=self.vectorizer.fit_transform(text)
        y=self.mlb.fit_transform(labels)
        self.model.fit(X,y)
        self.is_fitted=True
    
    def predict(self,test_text,labels):
        X_test=self.vectorizer.fit_transform(test_text)
        y_predt=self.model.predict(X_test)

        return self.mlb.inverse_transform(y_predt)
    
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