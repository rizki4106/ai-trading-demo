import streamlit as st
from PIL import Image
import pandas as pd
from core.model import initialize_model_v1
from core.controller import predict, device_diagnostic

# device diagnostic
device = device_diagnostic()

# initialize trained model
class_name = {0: "down", 1: "up"}
model_path = "./model/model-1.pth"
model = initialize_model_v1(device=device, saved_weight_path=model_path)

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
st.markdown("untuk melakukan prediksi silahkan upload gambar `chart pattern`")

# upload image
uploaded_image = st.file_uploader("Upload Gambar Chart Pattern", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:

    # read image
    img = Image.open(uploaded_image)

    # convert channel
    img = img.convert("RGB")

    # lakukan prediksi disini

    prob, classes = predict(model=model, image=img, device=device)

    st.image(img)
    st.markdown(f"File `{uploaded_image.name}` terdeteksi sebagai trend `{class_name[classes]}` dengan probabilitas `{prob:.5f}`")