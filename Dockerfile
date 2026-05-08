FROM python:3.10-slim

LABEL maintainer="Ibrahim Yetenek"
LABEL description="SpectralIF — ADS-B RF Signal Anomaly Detection (DS570 Final Project)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/healthz')" || exit 1

CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]