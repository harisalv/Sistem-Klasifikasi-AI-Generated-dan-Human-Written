# Sistem-Klasifikasi-AI-Generated-dan-Human-Written
# AI vs Human Text Classification Using BERT with LIME and SHAP

Repository ini berisi source code untuk implementasi klasifikasi teks **AI-generated** dan **Human-written** menggunakan model **BERT** dengan metode interpretabilitas **LIME** dan **SHAP**.

## Dataset

Dataset tidak disertakan di repository ini karena ukuran file yang cukup besar.
Silakan unduh dataset melalui Google Drive berikut:

**Dataset:**  
<[LINK_DATASET](https://drive.google.com/file/d/1RPWexiBTpCGNha_JlTk7VXvVgst0Uft7/view?usp=sharing)>

Setelah diunduh, letakkan file dataset pada direktori utama (root) project.

---

## Pre-trained Model

Model hasil fine-tuning (`best_model`) juga tidak disertakan di repository karena ukuran file yang besar.

Silakan unduh model melalui Google Drive berikut:

**Model:**  
<[LINK_MODEL](https://drive.google.com/drive/folders/15ieqai3tBhA8DCKc8iLAfAw5-Uxrwz0C?usp=sharing)>

Setelah diunduh, ekstrak folder sehingga struktur project menjadi seperti berikut:

```
WEB DEMO/
│
├── best_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── app.py
├── requirements.txt
└── ...
```

Pastikan dataset dan folder `best_model` telah ditempatkan pada lokasi yang sesuai sebelum menjalankan aplikasi.
SEMANGAT DIK ADIK MASA DEPAN
