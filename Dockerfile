FROM python:3.12-slim

LABEL maintainer="MuzikRE Team"
LABEL description="MuzikRE — Music Popularity Prediction"

# System deps for librosa (libsndfile) and audio processing (ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py audio_analyzer.py ./
COPY model/ model/
COPY data/ data/
COPY templates/ templates/
COPY static/ static/

# Copy utility scripts (optional, for retraining inside container)
COPY train_model.py collect_data.py expand_data.py ./

EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]
