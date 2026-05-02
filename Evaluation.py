from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(name, model, X_test, y_test):
    if name == "Isolation Forest":
        preds = model.predict(X_test)
        preds = [1 if p == -1 else 0 for p in preds]
        prob = preds
    else:
        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    try:
        auc = roc_auc_score(y_test, prob)
    except:
        auc = 0

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    }
