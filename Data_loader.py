import pandas as pd
from sklearn.datasets import make_classification

def load_synthetic_data():
    # Synthetic dataset similar to fraud scenario
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],  # Imbalance
        random_state=42
    )

    df = pd.DataFrame(X)
    df['target'] = y
    return df
