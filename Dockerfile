FROM python:3.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ./vendor/openenv-core && \
    pip install --no-cache-dir -r requirements.txt

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]