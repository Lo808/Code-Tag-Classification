import re


def clean_text_description(text: str) -> str:

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

def clean_text_code(text: str) -> str:
    """
    For cleaning the code, we keep punctuation because it carries meaning
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove unicode

    return text.strip()

    
def preprocess_data(data,use_code=False):
    '''
    Apply the cleaning function and convert to list
    :param data: data set
    :param use_code: Use the formal code as text if True
    '''

    description=data['prob_desc_description'].apply(clean_text_description)
    code=data['source_code'].apply(clean_text_code)

    if use_code:
        full_text=description+" "+code

    full_text=description

    labels=data['tags'].tolist()

    return full_text.tolist(),labels, data


