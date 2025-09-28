import joblib
import pandas as pd
from pathlib import Path

def test_model_file_exists():
    model_path = Path("models/iris_rf_model.pkl")
    assert model_path.exists(), "Model dosyası bulunamadı!"

def test_model_accuracy():
    model_path = Path("models/iris_rf_model.pkl")
    data_path = Path("data/processed/iris_processed.csv")


    model = joblib.load(model_path)  # kaydettiğimiz modeli burada yüklüyoruz
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target", "target_name"])
    y = df["target"]

   
    acc = model.score(X, y)

    
    assert acc > 0.7, f"Accuracy çok düşük: {acc}" #burada min başarı beklentisi tanımlıyoruz
