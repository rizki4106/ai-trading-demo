FROM python:3.9.16

WORKDIR /trading-app

COPY  . .

RUN pip3 install -r requirements.txt

CMD streamlit run model_1.py --server.port $PORT --server.headless true --server.enableCors false