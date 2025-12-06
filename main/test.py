from data_loader import load_json_data
from preprocessing import preprocess_data 
from baseline import tf_idf
from evaluation import compute_f1_scores, per_tag_f1
from sklearn.model_selection import train_test_split



data=load_json_data(r"C:\Users\maloc\OneDrive\Documents\Tag Classification\Code-Tag-Classification\data\code_classification_dataset.zip")


full_text,labels, data=preprocess_data(data)
train_text, test_text, train_labels, test_labels = train_test_split(
    full_text, 
    labels, 
    test_size=0.3,
    random_state=42,
    shuffle=True
)


model_tf=tf_idf()
model_tf.fit(train_text,train_labels)









