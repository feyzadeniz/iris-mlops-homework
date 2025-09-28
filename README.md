# Iris MLOps Homework

Bu proje, Iris veri seti üzerinde uçtan uca bir MLOps pipeline örneğidir.
DVC ile veri ve işlem takibi, scikit-learn ile model eğitimi, pytest ile test süreçleri otomatikleştirilmiştir.

# 🚀 Pipeline Aşamaları

Download Data → src/download_data.py

Iris veri setini indirir ve data/raw/iris.csv dosyasına kaydeder.

Clean Data → src/clean_data.py

Kolon adlarını düzenler, eksik değerleri kontrol eder, data/processed/iris_processed.csv üretir.

Train Model → src/train_model.py

Random Forest ile modeli eğitir, models/iris_rf_model.pkl ve metrikleri (metrics.json) kaydeder.

Test Model → tests/test_model.py

Pytest ile modelin ve çıktıların varlığını/performansını doğrular.

Test raporu reports/pytest.xml içine kaydedilir.

# Kullanım
# Sanal ortamı aktif et
source iris-venv/Scripts/activate

# Gereken paketleri yükle
pip install -r requirements.txt

# Pipeline'ı çalıştır
dvc repro

# Metrikleri görüntüle
dvc metrics show

# Testleri çalıştır
pytest -q tests/

