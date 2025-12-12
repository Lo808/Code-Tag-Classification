from sklearn.metrics import f1_score

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


def per_tag_f1(y_true, y_pred_binary, mlb_classes, focus_tags=None): 
    '''
    Docstring for per_tag_f1 
    :param y_true: matrix (n_samples, n_labels)
    :param y_pred_binary: matrix (n_samples, n_labels)
    :param mlb_classes: list of ALL label names (model.mlb.classes_)
    :param focus_tags: list of specific tags to print
    '''

    # compute f1 scores
    f1s=f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    
    # Store in dict {tag name : f1 score}
    scores_dict={tag: round(score, 2) for tag, score in zip(mlb_classes,f1s)} # type: ignore
    
    # get only focus tags
    if focus_tags:
        results=[]

        for tag in focus_tags:
            if tag in scores_dict:
                results.append((tag,scores_dict[tag]))

            else:
                print(f"Warning: Tag '{tag}' not found in model classes.")
        return results
    
    return list(scores_dict.items())

from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def compute_detailed_metrics(y_true, y_pred_binary):
    """
    Calcule F1, mais aussi Jaccard (plus intuitif pour le multi-label)
    et sépare Précision/Rappel.
    """
    
    # 1. F1 Scores (Tes métriques actuelles)
    micro_f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    
    # 2. Jaccard Score (Spécifique au Multi-Label)
    # average='samples' calcule le score pour chaque ligne puis fait la moyenne
    jaccard = jaccard_score(y_true, y_pred_binary, average="samples", zero_division=0)
    
    # 3. Diagnostic (Pourquoi le F1 est bas ?)
    # Precision : Est-ce que je dis n'importe quoi ?
    precision = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    # Recall : Est-ce que je rate des choses ?
    recall = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)

    return {
        "micro_f1": round(micro_f1, 3),"macro_f1": round(macro_f1, 3),"jaccard_score": round(jaccard, 3),"precision_macro": round(precision, 3),"recall_macro": round(recall, 3)} # type: ignore