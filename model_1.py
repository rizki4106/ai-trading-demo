import streamlit as st
from PIL import Image
import pandas as pd
# from core.model import initialize_model_v1
from core.controller import device_diagnostic
# from core.data import visualize_result
import yfinance as yf
import plotly.graph_objects as go

# new code
from core.ai import ConfirmModelV1
from core.tester import TraderTester

# device diagnostic
device = device_diagnostic()

# get live data
# get data 
ticker = yf.Ticker('BTC-USD')
data = ticker.history(period="7d", interval="15m")

# initialize trained model
# initialize model
m1 = ConfirmModelV1(state_dict_path="./model/model-1.pth",
                    device=device_diagnostic(),
                    class_name={0: "sell", 1 : "buy"})

# initialize tester
tester = TraderTester(
    historical_data=data.iloc[-50:, :],
    n_candle=12,
    model=m1,
)

# content
st.sidebar.markdown("# Model 1")
st.sidebar.markdown("dilatih menggunakan data `up` dan `down`")

# informasi data yang digunakan
st.markdown("# Model 1")
st.markdown("model 1 dilatih dengan menggunakan 2 data yaitu data `up` dan `down`")

# informasi performa model
st.markdown("## Performa")
st.markdown("berikut ini adalah grafik `loss/error` dan `akurasi`")

# membaca gambar aperforma
performa = Image.open('./data/model-1-performance.png')

# informasi performa
info = pd.DataFrame([{
    "akurasi": "91%",
    "loss/error": "0.35346"
}])

info.reset_index()

st.dataframe(info)
st.image(performa)

# inference mode / predksi
st.markdown("## Prediksi")

# upload image
uploaded_image = st.file_uploader("Upload Gambar Chart Pattern", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:

    # read image
    img = Image.open(uploaded_image)

    # convert channel
    img = img.convert("RGB")

    # lakukan prediksi disini

    res = m1.predict(image=img)

    st.image(img)
    st.markdown(f"File `{uploaded_image.name}` terdeteksi sebagai trend `{res['class_name']}` dengan probabilitas `{res['probability']:.5f}`")


st.markdown("## Backward Test")

with st.spinner("mohon tunggu sedang melakukan backward test..."):
    

    st.write("BTC-USD")
    st.write("period `7d` interval `15m`")

    # create annotation
    annotation = tester.back_test()

    data = data.iloc[-50:, :-2]
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    fig.update_layout(annotations=annotation, height=800)

    st.plotly_chart(fig)