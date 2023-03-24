import streamlit as st
from PIL import Image

st.sidebar.markdown("# Model 2")
st.sidebar.markdown("dilatih menggunakan data `up benar`, `down benar`, `up salah` dan `down salah`")

# informasi data yang digunakan
st.markdown("# Model 1")
st.markdown("model 1 dilatih dengan menggunakan 2 data yaitu data `up` dan `down`")

# informasi performa model
st.markdown("## Performa")
st.markdown("berikut ini adalah grafik `loss/error` dan `akurasi`")

# inference mode / predksi
st.markdown("## Prediksi")
st.markdown("untuk melakukan prediksi silahkan upload gambar `chart pattern`")

# upload image
uploaded_image = st.file_uploader("Upload Gambar Chart Pattern", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:

    # read image
    img = Image.open(uploaded_image)

    st.image(img)