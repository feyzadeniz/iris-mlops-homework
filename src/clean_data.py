import pandas as pd
from pathlib import Path

RAW = Path("data/raw/iris.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED_DIR / "iris_processed.csv"

def main():
    df = pd.read_csv(RAW)  #veriyi referanstan okuduk
    df.columns = [col.lower() for col in df.columns] #kolon adlarını küçük harfe çevirdik
    missing = df.isnull().sum() #eksik değer kontrolü
    print(f"Kolon bazında eksik değerler: {missing}")

    # eksik değer varsa diye
    df = df.fillna(df.mean(numeric_only=True)) #ortalama ile

    # kategorik değişken oluşturmak için
    target_map = {0: "setosa", 1: "versicolor", 2: "virginica"} #sözlük oluşturduk
    df["target_name"] = df["target"].map(target_map) # 0,1li değerleri target isimleriyle değiştirip yeni kolon haline getirdik

    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Veri temizlendi ve kaydedildi: {OUT} (shape={df.shape})")

    if __name__ == "__main__":
        main()

