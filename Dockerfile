FROM python:3.12-slim
# ffmpeg grabs and resizes camera frames.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vlm_on_pi ./vlm_on_pi
ENV DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "vlm_on_pi.web:app", "--host", "0.0.0.0", "--port", "8000"]
