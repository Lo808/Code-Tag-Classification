import re
from sklearn.model_selection import train_test_split
from collections import Counter


class Preprocessor():

    def __init__(self,use_code=False) -> None:
        self.use_code=use_code
        pass

    def clean_text_description(self,text: str) -> str:

        """
        Basic cleaning of problem descriptions, removing recurent meaningless characters  and Latex commands. 
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        
        text = re.sub(r"(\$+)", "", text)         # remove $$$ latex formatting
        text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove unicode
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
        :param use_code: Use the formal code as text if True
        '''

        description=data['prob_desc_description'].apply(self.clean_text_description)
        code=data['source_code'].apply(self.clean_text_code)
        

        if self.use_code:
            full_text=description+" "+code
        else:
            full_text=description

        labels=data['tags'].tolist()

        return full_text.tolist(),labels, data
    

    def process_data(self,data,devset=True,test_size=0.3,check_balance=True):
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

            
            self.check_split_balance(
                train_labels, dev_labels, test_labels,
                focus_tags=['math','graphs','strings','number theory',
                            'trees','geometry','games','probabilities']
            )
            return train_text, dev_text, test_text, train_labels, dev_labels, test_labels
        
        
        self.check_split_balance(train_labels, None, test_labels)
        
        return train_text, test_text, train_labels, test_labels
    
    def check_split_balance(self, train_labels, dev_labels=None, test_labels=None, focus_tags=None):
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
        if focus_tags:
            print("\n=== Focus Tags Statistics ===")
            for tag in focus_tags:
                print(
                    f"{tag:15s} | train: {train_count.get(tag,0):3d} "
                    f"dev: {dev_count.get(tag,0):3d} "
                    f"test: {test_count.get(tag,0):3d}"
                )
        print("TRAIN SIZE:", len(train_labels))
        print("DEV SIZE:", len(dev_labels) if dev_labels else None)
        print("TEST SIZE:", len(test_labels) if test_labels else None)

    
 .
