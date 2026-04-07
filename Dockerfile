FROM python:3.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]