import re
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import unicodedata


class Preprocessor():
    
    def __init__(self,use_code=False) -> None:
        self.use_code=use_code
        self.mlb=MultiLabelBinarizer()
        self.label_order=None
        self.focus_tags=['math','graphs','strings','number theory','trees','geometry','games','probabilities']
        pass

    def clean_text_description(self,text: str) -> str:

        """
        Basic cleaning of problem descriptions, removing recurent meaningless characters  and Latex commands. 
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        
        text = re.sub(r"(\$+)", "", text)         # remove $$$ latex formatting
        text = unicodedata.normalize("NFKC", text) # normalize unicode
        text = re.sub(r'\\[a-zA-Z]+', ' ', text) # remove LaTeX commands like \frac, \le, \dots
        text = re.sub(r'[^\w\s]', ' ', text)     # replace punctuation with spaces
        text = re.sub(r"\s+", " ", text)          # normalize whitespace
        return text.strip()

    def clean_text_code(self,text: str) -> str:
        """
        For cleaning the code, we keep punctuation because it carries meaning
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"\s+", " ", text)          # normalize whitespace
        text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove unicode

        return text.strip()
   
    def apply_cleaning(self,data):
        '''
        Apply the cleaning function and convert to list
        :param data: data set
        '''

        description=data['prob_desc_description'].apply(self.clean_text_description)
        code=data['source_code'].apply(self.clean_text_code)
        

        if self.use_code:
            full_text=description+"[CODE]"+ code
        else:
            full_text=description

        labels=data['tags'].tolist()

        return full_text.tolist(),labels, data
    
    def process_data(self,data,devset=True,test_size=0.3,binarize_labels=True,check_balance=True):
        '''
        Docstring for process_data
        
        :param data: raw data
        :param devset: split with a devset or not, split is always at 50% of test_size
        :param test_size: correpond to the proportion of (dev+test) set or just test set
        :param binarize_labels: Label output already binarized using the MLB or not
        :param check_balance: Print a table of size for each important classes in all set to check for presence/balance
        '''

        # Apply cleaning with previous methods
        full_text,labels,_=self.apply_cleaning(data)

        train_text, test_text, train_labels, test_labels = train_test_split(
        full_text, 
        labels, 
        test_size=test_size,
        random_state=42,
        shuffle=True)


        if devset==True:
            dev_text, test_text, dev_labels, test_labels = train_test_split(
                test_text,
                test_labels,
                test_size=0.50,      # half for dev, half for test
                random_state=42,
                shuffle=True
            )

            
            self.check_split_balance(train_labels,dev_labels,test_labels)

            train_labels_bin,test_labels_bin,dev_labels_bin=self.binarize_labels(train_labels,test_labels,dev_labels) # type: ignore

            return train_text, dev_text, test_text, train_labels_bin, dev_labels_bin, test_labels_bin
        

        if check_balance:
            self.check_split_balance(train_labels, None, test_labels)

        train_labels_bin,test_labels_bin=self.binarize_labels(train_labels,test_labels,dev_labels=None) # type: ignore
        
        return train_text, test_text, train_labels_bin, test_labels_bin
    
    def binarize_labels(self,train_labels,test_labels,dev_labels=None):

        self.mlb.fit(train_labels)
        self.label_order=list(self.mlb.classes_)
        train_labels_bin=self.mlb.transform(train_labels)
        test_labels_bin=self.mlb.transform(test_labels)

        if dev_labels is not None:
            dev_labels_bin=self.mlb.transform(dev_labels)
            return train_labels_bin,test_labels_bin,dev_labels_bin
        
        
        return train_labels_bin,test_labels_bin


    def check_split_balance(self, train_labels, dev_labels=None, test_labels=None):
        """
        Check class balance across splits for multilabel data.
        Prints useful diagnostics.
        """

        def count_tags(labels):
            return Counter(tag for sample in labels for tag in sample)

        train_count = count_tags(train_labels)
        dev_count = count_tags(dev_labels) if dev_labels else Counter()
        test_count = count_tags(test_labels) if test_labels else Counter()

        print("\n=== CHECKING SPLIT BALANCE ===")

        # ---- Missing tags check ----
        if dev_labels:
            dev_missing = set(dev_count) - set(train_count)
            if dev_missing:
                print("⚠️ Tags in DEV but not in TRAIN:", dev_missing)

        if test_labels:
            test_missing = set(test_count) - set(train_count)
            if test_missing:
                print("⚠️ Tags in TEST but not in TRAIN:", test_missing)

        # ---- Focus tag counts ----
        
            print("\n=== Focus Tags Statistics ===")
        for tag in self.focus_tags:
            print(
                f"{tag:15s} | train: {train_count.get(tag,0):3d} "
                f"dev: {dev_count.get(tag,0):3d} "
                f"test: {test_count.get(tag,0):3d}"
                )
        print("TRAIN SIZE:", len(train_labels))
        print("DEV SIZE:", len(dev_labels) if dev_labels else None)
        print("TEST SIZE:", len(test_labels) if test_labels else None)

    

class Tensorizer:
    '''
    Tensorizing Class to encode text into tensorized data
    '''
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def convert_text_to_tensor(self, text):
        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten()
        }
    

class CodeforcesDataset(Dataset):
    '''
    Dataset subclass to process data to model
    '''
    def __init__(self, texts, labels, tensorizer):
        """
        texts: Liste de textes bruts
        labels: Liste de labels
        tensorizer: Une instance de Tensorizer
        """
        self.texts = texts
        self.labels = labels
        self.tensorizer = tensorizer 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        text = self.texts[item]
        label = self.labels[item]

        tensors = self.tensorizer.convert_text_to_tensor(text)
        tensors['labels'] = torch.tensor(label, dtype=torch.float)
        
        return tensors
    
