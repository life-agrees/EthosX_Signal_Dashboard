# api_server/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1) Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy source code
COPY . .

# 3) Expose the port your app listens on
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# 4) Optional: add a healthcheck so orchestrators know when
#    your backend is up (make sure you have a /health or /ping endpoint)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# 5) Use shell form so the ENV vars actually get substituted
ENTRYPOINT ["sh","-c"]
CMD ["uvicorn api_server:app --host $HOST --port $PORT"]
