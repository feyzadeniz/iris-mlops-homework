from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path  #dosya klasör yollarını yönetmek için


RAW_DIR = Path("data/raw")  # klasörü temsil eden Path nesnesi
RAW_DIR.mkdir(parents=True, exist_ok=True) #klasör yoksa oluşturur, varsa hata verdirmez
OUT = RAW_DIR / "iris.csv" #bu csv o adrese kaydedilecek
# bu kısım os ile de yapılabiliyormuş

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(OUT, index=False, encoding="utf-8") #OUT yoluna kaydet, satır no yazma
    print(f"Iris verisi kaydedildi: {OUT} (shape = {df.shape})")

    
if __name__ == "__main__":
     main()
