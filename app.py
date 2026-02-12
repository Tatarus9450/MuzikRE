"""
Music Popularity Regression - Flask Web Application
=====================================================
Serves a prediction web application where users can input
music features and get a predicted popularity score.

Endpoints:
  GET  /              → Serve the main prediction page
  POST /predict       → Accept JSON features, return prediction
  POST /predict-file  → Upload audio file, extract features, predict
  GET  /model-info    → Return model evaluation metrics
"""

import os
import json
import tempfile
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from audio_analyzer import analyze_audio

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma', 'aiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model and scaler at startup
MODEL_DIR = "model"
model = joblib.load(os.path.join(MODEL_DIR, "regression_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

# Load evaluation metrics
with open(os.path.join(MODEL_DIR, "evaluation.json"), "r") as f:
    evaluation = json.load(f)

FEATURES = ['duration_min', 'tempo_bpm', 'energy', 'danceability', 'loudness_db']

FEATURE_RANGES = {
    'duration_min': {'min': 1.5, 'max': 8.0, 'step': 0.1, 'default': 3.5},
    'tempo_bpm':    {'min': 60,  'max': 200, 'step': 1,   'default': 120},
    'energy':       {'min': 0.0, 'max': 1.0, 'step': 0.01,'default': 0.6},
    'danceability': {'min': 0.0, 'max': 1.0, 'step': 0.01,'default': 0.6},
    'loudness_db':  {'min': -30, 'max': 0,   'step': 0.5, 'default': -8},
}


@app.route("/")
def index():
    """Serve the main prediction page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict music popularity from input features."""
    try:
        data = request.get_json()
        
        # Validate and extract features
        values = []
        for feat in FEATURES:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400
            val = float(data[feat])
            r = FEATURE_RANGES[feat]
            val = max(r['min'], min(r['max'], val))
            values.append(val)
        
        # Predict
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        # Clip to valid range
        prediction = float(np.clip(prediction, 0, 100))
        
        return jsonify({
            "popularity": round(prediction, 1),
            "features": dict(zip(FEATURES, values)),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-file", methods=["POST"])
def predict_file():
    """Upload audio file, extract features, and predict popularity."""
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "ไม่พบไฟล์เสียง กรุณาอัปโหลดไฟล์"}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({"error": "ไม่ได้เลือกไฟล์"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"ไม่รองรับไฟล์ประเภทนี้ รองรับเฉพาะ: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Save to temp file
        ext = file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract features from audio
            features = analyze_audio(tmp_path)
            
            # Predict
            values = [features[f] for f in FEATURES]
            X = np.array(values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            prediction = float(np.clip(prediction, 0, 100))
            
            return jsonify({
                "popularity": round(prediction, 1),
                "features": features,
                "file_name": file.filename,
                "status": "success"
            })
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาด: {str(e)}"}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model evaluation metrics and info."""
    return jsonify({
        "evaluation": evaluation,
        "feature_ranges": FEATURE_RANGES,
        "features": FEATURES,
        "model_type": "Multiple Linear Regression",
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })


if __name__ == "__main__":
    print("🎵 Music Popularity Regression - Web App")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
            host="0.0.0.0", port=5000)
