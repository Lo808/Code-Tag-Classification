from sklearn.metrics import f1_score
import numpy as np


def compute_f1_scores(y_true,y_pred_binary) -> dict:
    """
    Computes micro and macro F1 scores.
    Args:
        y_true: numpy array (n_samples, n_labels)
        y_pred_binary: numpy array (n_samples, n_labels)
        labels: optional list of tag names (for printing)
    """

    micro = f1_score(y_true,y_pred_binary,average="micro", zero_division=0)
    macro = f1_score(y_true,y_pred_binary,average="macro", zero_division=0)

    return {"micro_f1": micro,"macro_f1":macro}


def per_tag_f1(y_true,y_pred_binary,tag_names): 
    """
    Computes F1 score for each tag (useful for the 8 target tags).
    Args:
        y_true: matrix (n_samples, n_labels)
        y_pred_binary: matrix (n_samples, n_labels)
        tag_names: list of label names (from MultiLabelBinarizer)
    """

    f1s=f1_score(y_true,y_pred_binary,average=None,zero_division=0)
    
    per_tag_results = list(zip(tag_names,f1s))
    return per_tag_results