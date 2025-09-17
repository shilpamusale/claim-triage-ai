# claimtriageai/utils/functions.py

import pickle
import joblib
from typing import Any

def convert_to_int(x: Any) -> Any:
    return x.astype(int) if hasattr(x, "astype") else x

def load_pickle(path: Any) -> Any:
    path_str = str(path)
    if path_str.endswith(".joblib"):
        return joblib.load(path_str)
    with open(path_str, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    data = load_pickle("models/denial_prediction_model.joblib")
    print(type(data))