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