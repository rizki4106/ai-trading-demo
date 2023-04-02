import streamlit as st
import yfinance as yf
from core.ai import CandleStick, ConfirmModelV1, SignalConfirmV1
from core.visualizer import create_annotation_v2, annotate
from core.data import create_image
import plotly.graph_objects as go

# pengaturan sidebar
st.sidebar.markdown("# Model 3")
st.sidebar.markdown("Gabungan antara model 1 dan 2")

# content
st.markdown("# Model 3")
st.markdown("Model ini menggunakan gabungan antara model 1 (candle stick 12 bar) sebagai pendeteksi sinyal dan model 2 (turunan `IF THEN`) sebagai sinyal konfirmasi")

st.markdown("## Back Test")

# get data
ticker = yf.Ticker('EURUSD=X')
data = ticker.history(period="7d", interval="15m").iloc[-50:, :]

# inisialisasi model
cs = CandleStick()
signal = ConfirmModelV1(
    state_dict_path='./model/model-1.pth',
    device='cpu',
    class_name={0: "sell", 1: "buy"}
)
confirm = SignalConfirmV1(signal, cs, 'cpu')

with st.spinner("Menjalankan prediksi..."):

    # define n_candle for confirmation as signal model
    n_candle_signal = 12

    # define n_candle for candlestick as confirmation signal
    n_candle_confirmation = 3

    # define annotation point
    annt = []

    # define result variable
    result = []

    for i in range(len(data)):

        # select 12 the last signal
        selected_signal = data.iloc[i:n_candle_signal, :]

        # select 3 candle from selected signal
        selected_candle = selected_signal.iloc[-3:, :]

        # create image from candle
        image = create_image(selected_candle)

        # run prediction
        preds = confirm.predict(image, selected_candle)

        if preds['class_int'] == 1 or preds['class_int'] == 0:
            annt.append(n_candle_signal)
            result.append(preds['class_int'])
        else:
            pass

        # update n_candle by 1
        n_candle_signal += 1
    
    # create annotation
    annt_point = annotate(annt, result, data, signal.class_name)

    st.markdown("EURUSD=X periode `7 hari` rentang waktu `15 menit`")

    # plot candle stick kedalam chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    fig.update_layout(annotations=annt_point, height=800)

    st.plotly_chart(fig)

