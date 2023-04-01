import streamlit as st
from PIL import Image
from core.ai import ConfirmModelV1, SignalModelV2
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from core.data import create_image
from core.ai import CandleStick
from core.visualizer import annotate

# inisialisasi model benar
m1 = ConfirmModelV1(state_dict_path="./model/model-3-true-signal.pth",
                    device="cpu",
                    class_name={0: "sell", 1 : "buy"})
# inisialisasi model salah
m2 = ConfirmModelV1(state_dict_path="./model/model-3-false-signal.pth",
                    device="cpu",
                    class_name={0: "sell", 1 : "buy"})
# initialisasi model gabungan
model = SignalModelV2(model_true=m1, model_false=m2, device="cpu")

# inisialisasi model konfirmasi
cs = CandleStick()

# pengaturan sidebar
st.sidebar.markdown("# Model 4")
st.sidebar.markdown("Model 4 menggunakan 2 model yaitu model yang dilatih dengan data benar dan model yang dilatih dengan data salah")

# content
st.markdown("# Model 4")
st.markdown("Model 4 menggunakan 2 model yaitu model yang dilatih dengan data benar dan model yang dilatih dengan data salah")

st.markdown("# Performa")

st.markdown("##### 1. Performa Model yang Dilatih dengan Data Benar")
st.markdown("**Akurasi**: 95%")
st.markdown("**Loss**: 0.20")
st.image(Image.open('./data/performance-true-acc-95-loss-0.20.png'), 'performa model yang dilatih dengan data benar')

st.markdown("##### 2. Performa Model yang Dilatih dengan Data Salah")
st.markdown("**Akurasi**: 95%")
st.markdown("**Loss**: 0.28")
st.image(Image.open('./data/performance-false-acc-95-loss-0.28.png'), 'performa model yang dilatih dengan data salah')

st.markdown("# Prediksi")
# upload image
# kemudian lakukan prediksi
uploaded_image = st.file_uploader("Upload Gambar Chart Pattern", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:

    with st.spinner("Memprediksi..."):
        # read image
        img = Image.open(uploaded_image)

        # convert channel
        img = img.convert("RGB")

        # lakukan prediksi disini

        res = model.predict(image=img)

        st.image(img)

        df = pd.DataFrame(res)

        st.table(df)

        st.markdown("Kesimpulan")

        if res[0]['probability'] > res[1]['probability']:
            st.markdown(f"File `{uploaded_image.name}` terdeteksi sebagai trend `{res[0]['class_name']}` dengan probabilitas `{res[0]['probability']:.5f}`")
        else:
            st.markdown(f"Tidak melakukan aksi karena *probabilitas salah `({res[1]['probability']:.5f})` **>** probabilitas benar `({res[0]['probability']:.5f})`*")

st.markdown("# Back Test")
st.markdown("Data live `BTC-USD`")

# mengambil data euro usd
# periode 7 hari dan rentang waktu 15 menit
data = yf.download("BTC-USD", period="7d", interval="15m")

# mengambil 50 data terakhir
data = data.tail(50)

# melakukan prediksi untuk setiap 12 data
# kemudian iterasi selanjutnya n_candle + 1
# sampai data terakhir
with st.spinner("Melakukan back test..."):

    result = []
    annt = []
    n_data = 12

    r  = []

    for i in range(0, len(data)):

        try:

            # mengambil data 12 data
            sample = data.iloc[i:n_data, :-2]

            # membuat gambar dari data tersebut
            img = create_image(sample)

            # cek sinyal konfirmasi menggunakan 3 data candle terakhir
            candle = sample.iloc[-3:, :]

            # prediksi sinyal konfirmasi
            preds = cs.do_transaction(candle)

            # prediksi pattern yang membangun sinyal konfirmasi
            res = model.predict(image=img)

            # if res[0]['probability'] > res[1]['probability']:

            #     if preds == res[0]['class_int']:
            if preds != 2:
                

                final = {
                    "konfirmasi": preds,
                }

                if res[0]['probability'] > res[1]['probability']:
                    final["sinyal"] = res[0]['class_int']
                    final["probability"] = res[0]['probability']
                    final['model'] = "benar"

                    if preds == res[0]['class_int']:
                        result.append(preds)
                        annt.append(i+12)
                else:
                    final["sinyal"] = res[1]['class_int']
                    final["probability"] = res[1]['probability']
                    final['model'] = "salah"
                r.append(final)

            
        except:
            pass
            
        n_data += 1

    st.table(r)
    # menampilkan chart menggunakan plotly
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
    fig.update_layout(annotations=annotate(annt, result, data, {0: "sell", 1 : "buy"}), height=800)

    st.plotly_chart(fig)

