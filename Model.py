from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_models():
    models = {}

    # Cost-sensitive learning using class_weight
    models['Logistic Regression'] = LogisticRegression(class_weight='balanced', max_iter=1000)

    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    models['SVM'] = SVC(kernel='rbf', class_weight='balanced', probability=True)

    # Isolation Forest (unsupervised)
    models['Isolation Forest'] = IsolationForest(contamination=0.05, random_state=42)

    return models
