FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py ./models.py
COPY simulation.py ./simulation.py
COPY environment.py ./environment.py
COPY server/app.py ./app.py
COPY openenv.yaml ./openenv.yaml

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
