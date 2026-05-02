from data_loader import load_synthetic_data
from preprocessing import preprocess_data
from models import get_models
from evaluation import evaluate_model
from utils import print_results

def main():
    print("Loading data...")
    df = load_synthetic_data()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training models...")
    models = get_models()

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "Isolation Forest":
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)

        res = evaluate_model(name, model, X_test, y_test)
        results.append(res)

    print_results(results)


if __name__ == "__main__":
    main()
