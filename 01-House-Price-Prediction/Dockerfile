FROM python:3.11-slim

# Install libgomp1 — required by scikit-learn / xgboost in slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching — only re-installs deps if they change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
    
# Copy the rest
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Non-root user 
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]