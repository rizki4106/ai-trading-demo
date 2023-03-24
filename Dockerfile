FROM python:3.11

WORKDIR /trading-app

COPY  . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD streamlit run model_1.py --server.port $PORT
