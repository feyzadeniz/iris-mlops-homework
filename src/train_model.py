from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = Path("data/processed/iris_processed.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("target", "target_name")]
    X = df[feature_cols]
    y = df["target"]  # X/y ayırdık

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,  # işlemi kaç cpu çekirdeği lullanarak yapacağını belirleyen param. -1 bilgisayardaki tüm çekirdekleri kullanır
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "macro avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall": round(report["macro avg"]["recall"], 4),
            "f1-score": round(report["macro avg"]["f1-score"], 4),
        },
        "weighted avg": {
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall": round(report["weighted avg"]["recall"], 4),
            "f1-score": round(report["weighted avg"]["f1-score"], 4),
        },
    }

    # macro - sınıf dengesizliklerini umursamıyormuş, weighted ağırlıklı ortalama olduğu için dengesizliklere duyarlıymış
    
    model_path = MODEL_DIR / "iris_rf_model.pkl"
    joblib.dump(model, model_path) # bunu yaparsak daha sonra başka bi dosyada yükleyip kullanabiliriz

    # metrikleri kaydediyoruz
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


    print(f"Model kaydedildi: {model_path}")
    print(f"Metrikler: {metrics_path}")
    print(json.dumps(metrics, indent=2)) 

if __name__ == "__main__":
    main()




