# Weather Classification XGBoost

Project ini sudah disiapkan untuk deploy model klasifikasi cuaca berbasis XGBoost.

## Menjalankan di Lokal

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

Jika ingin tuning hyperparameter:

```powershell
python src/train_model.py --tune --n-iter 12
```

## Deploy Streamlit Community Cloud

1. Upload folder project ini ke GitHub.
2. Pastikan file `requirements.txt`, `app.py`, folder `models`, dan `weather_classification_data.csv` ikut tersimpan.
3. Di Streamlit Community Cloud, pilih repository ini.
4. Isi main file path dengan `app.py`.
5. Deploy.

## File Penting

- `src/train_model.py`: training pipeline XGBoost dan penyimpanan model.
- `models/weather_xgboost.joblib`: artefak model siap pakai oleh aplikasi.
- `models/metrics.json`: hasil evaluasi model.
- `app.py`: aplikasi Streamlit untuk prediksi cuaca.
