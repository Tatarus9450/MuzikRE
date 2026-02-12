# 🎵 MuzikRE — Predict Music Popularity with Machine Learning

> **"Where Sound Meets Science"** — A project analyzing and predicting music popularity using Multiple Linear Regression.

![MuzikRE Banner](static/images/banner.png) 
*(Optional banner placeholder)*

---

## 🌟 About the Project
**MuzikRE** is a web application that merges **Data Science** with **Music**. The system analyzes your uploaded audio files, extracts key audio features, and processes them through a mathematical model to predict: **"How popular will this song be?"** (Popularity Score 0-100).

## 🚀 Key Features
- **🎧 Upload & Auto-Predict:** Simply drag and drop an audio file → The system automatically analyzes features and predicts the result instantly.
- **🎛️ Manual Prediction:** Adjust feature sliders yourself (e.g., Tempo, Loudness) to experiment with how different values affect popularity.
- **📊 Interactive Visualization:** Results are displayed through beautiful, easy-to-understand gauges and charts.

---

## 🛠️ Tech Stack
This project is built using modern, industry-standard technologies:

| Category | Technology |
|---|---|
| **Backend** | 🐍 Python, Flask, Gunicorn |
| **Machine Learning** | 🧠 scikit-learn, joblib, numpy, pandas |
| **Audio Processing** | 🎼 librosa, soundfile |
| **Frontend** | 🎨 HTML5, CSS3 (Glassmorphism), JavaScript (ES6+), Web Audio API |
| **Deployment** | 🐳 Docker, Docker Compose |

---

## 📈 Model Performance
The current model uses **Multiple Linear Regression**, trained on over **1,144 real-world song samples** across various genres.

### Current Accuracy Metrics:
| Metric | Value | Meaning |
|---|---|---|
| **R² (Train)** | **0.4962** | The model explains approximately 49.6% of the variance in the training data. |
| **R² (Test)** | **0.3512** | The model explains approximately 35.1% of the variance in unseen test data. |
| **MAE** | **9.05** | The average prediction error is approximately ±9 points (out of 100). |
| **RMSE** | **11.53** | The root mean squared error of the predictions. |

> **Note:** Predicting music popularity is inherently subjective and depends on numerous external factors (marketing, artist fame, social trends) that this model may not fully capture. Therefore, a moderate R² score is expected for this type of problem.

### Key Factors Influencing Popularity (Feature Importance):
1. **Danceability (+5.90)**: The easier it is to dance to, the more likely it is to be popular. 💃
2. **Loudness (+6.06)**: Mastered tracks that are louder tend to have higher popularity scores. 🔊
3. **Duration (-2.21)**: Songs that are too long may see slightly reduced popularity. ⏱️

---

## 🏁 Quick Start

The easiest way to run the application is via **Docker**:

```bash
# 1. Build and run the container
docker compose up --build

# Or run in the background (detached mode)
docker compose up --build -d
```

Once running, open your browser and navigate to:
👉 **http://localhost:5000**

---

## 📂 Project Structure
```
MuzikRE/
├── app.py                 # Main Web Server (Flask)
├── audio_analyzer.py      # Audio Analysis System (The Core Engine 💖)
├── train_model.py         # Script for retraining the model
├── collect_data.py        # Script for generating initial data
├── expand_data.py         # Script for expanding dataset with realistic profiles
├── Dockerfile             # Docker Image build instructions
├── docker-compose.yml     # Docker Compose configuration
├── model/                 # Directory for model files (.joblib)
├── data/                  # Directory for datasets (.csv)
└── static/                # Frontend assets (CSS, JS, Images)
```

---

## 🧪 Re-training the Model
If you wish to improve the model or update it with new data:

1. **Expand Dataset (Optional):**
   ```bash
   python expand_data.py
   ```
2. **Train Model:**
   ```bash
   python train_model.py
   ```
   The system will automatically save the new model files to the `model/` directory.

---

Developed by **Talu Khulapwan**
