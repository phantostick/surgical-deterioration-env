FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout=60 -r requirements.txt

COPY models.py ./models.py
COPY simulation.py ./simulation.py
COPY environment.py ./environment.py
COPY server/app.py ./app.py
COPY openenv.yaml ./openenv.yaml

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]