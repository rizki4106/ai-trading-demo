FROM python:3.11

WORKDIR /trading-app

COPY  . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "model_1.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableWebsocketCompression=false"]
