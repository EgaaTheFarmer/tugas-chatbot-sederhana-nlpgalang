# Chatbot Medis - Symptom Checker

Aplikasi chatbot medis berbasis machine learning untuk mengidentifikasi kemungkinan penyakit berdasarkan deskripsi gejala.

## Teknologi

- **Model:** TF-IDF + Support Vector Machine (LinearSVC)
- **Dataset:** Symptom2Disease dari Kaggle (1.200 data, 24 penyakit)
- **Framework:** Streamlit
- **Bahasa:** Python

## Fitur

- Input gejala dalam Bahasa Indonesia
- Prediksi penyakit menggunakan model machine learning
- Pertanyaan lanjutan untuk mempersempit diagnosis
- Informasi penyakit dan saran umum

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur File

```
├── app.py                    # Aplikasi Streamlit
├── requirements.txt          # Dependencies
├── svm_model.joblib          # Model terlatih
├── tfidf_vectorizer.joblib   # TF-IDF vectorizer
├── label_encoder.joblib      # Label encoder
└── .streamlit/
    └── config.toml           # Konfigurasi Streamlit
```

## Catatan

Aplikasi ini dibuat untuk keperluan edukasi dan bukan pengganti diagnosis medis profesional.
