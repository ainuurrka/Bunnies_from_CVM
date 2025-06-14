import pandas as pd
from src.model import load_model
from src.config import MODEL_PATH

def main():
    df = pd.read_csv("data/validation_features.csv")
    X = df.drop(columns=["application_id"])
    app_ids = df["application_id"]

    model = load_model(MODEL_PATH)
    preds = model.predict_proba(X)[:, 1] > 0.5

    result = pd.DataFrame({
        "id": app_ids,
        "isFrod": preds.astype(bool)
    })
    result.to_csv("result.csv", index=False)

if __name__ == "__main__":
    main()
