FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip==24.2
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY backend.py .
COPY themes.json .
COPY .env .
COPY blender_addon.py .
RUN mkdir -p /app/templates /app/db && chmod -R 777 /app /app/db
CMD ["sh", "-c", "chmod 666 /app/db/scene_context.db && uvicorn backend:app --host 0.0.0.0 --port 8000"]