import streamlit as st
import yfinance as yf
from core.ai import CandleStick
from core.visualizer import annotate
import plotly.graph_objects as go

st.sidebar.markdown("# Model 2")
st.sidebar.markdown("Menggunakan turan `IF THEN` untuk memprediksi trend")


st.markdown("# Model 2")
st.markdown("Model ini menggunakan turan `IF THEN` untuk memprediksi trend")

# get data
ticker = yf.Ticker('BTC-USD')
data = ticker.history(period="7d", interval="15m").iloc[-150:, :]


# inisialisasi class candlestick
cs = CandleStick()
with st.spinner("Menjalankan prediksi..."):

    # menjalankan prediksi
    annotation, prediction = cs.test_candle_transaction(data)

    # membuat anotasi
    annt_point = annotate(annotation, prediction, data, cs.rule_name)

    st.markdown("BTC-USD periode `7 hari` rentang waktu `15 menit`")

    # plot candle stick kedalam chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    fig.update_layout(annotations=annt_point, height=800)

    st.plotly_chart(fig)