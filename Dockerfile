FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run train.py to generate the model during build time (optional, if model is not pushed to git)
# RUN python train.py

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
