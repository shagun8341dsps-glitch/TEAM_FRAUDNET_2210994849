import pandas as pd

def print_results(results):
    df = pd.DataFrame(results)
    print("\n===== MODEL PERFORMANCE =====")
    print(df.sort_values(by="F1", ascending=False))
