import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

def create_data_point(frame, n_candle):
  """
  Create end of data point this function used for plot trend for all data
  """
  # define annotaton point
  point = []

  # create annotation point
  for i in range(0, len(frame), n_candle):
    point.append(i)

  return point

def hitung_kemiringan(x1, x2, y1, y2):
  """Menghitung kemiringan garis
  (y2 - y1) / (x2 - x1)s
  Args:
    x1, x2, y1, y2
  """

  pembilang = y2 - y1

  penyebut = x2 - x1

  # handle division by zero
  if penyebut == 0.0:
    penyebut = 0.1
  
  hasil = pembilang / penyebut

  return hasil

def predict_slope(frame):
  """
    Predict line splope
    higher positif output it's mean trend is strong bulish
    higher negative output it's mean trend is strong bearish

    Args:
      frame : pandas dataframe -> OHLC data
    
    Returns:
      slope : float -> value is range from infinite negative to infinite positive
  """

  # define x1 coordinat point ( using index of the first data )
  x1 = 0

  # define x2 coordinat point ( using the last index of data point )
  x2 = len(frame) - 1

  # define y1 coordinat point ( using close price )
  y1 = frame.iloc[x1, 3]

  # define y2 coordinat poin( using close price )
  y2 = frame.iloc[x2, 3]

  # calculate slope
  kemiringan = hitung_kemiringan(x1, x2, y1, y2)

  return kemiringan

st.sidebar.markdown("# Model 5")
st.sidebar.markdown("Model 5 memprediksi trend dengan metode kemiringan")

st.markdown("# Model 5")
st.markdown("dibawah ini merupakan hasil dari prediksi trend sinyal menggunakan metode kemiringan. Semakin besar nilai `positif` itu artinya trend naik kuat dan semakin besar nilai `negative` itu artinya trend turun kuat.")
latext = r'''
#### Rumus Kemiringan
$$ 
kemiringan = \frac{y2 - y1}{x2 - x1}
$$ 
'''
st.write(latext)
# mendownload data
ticker = yf.download("BTC-USD", period="7d", interval="15m")
# define data that will use
data = ticker.iloc[-100:, :]

# create data point
point = create_data_point(data, 12)

# Membuat plot candlestick
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

for i in range(len(point)):
  if i + 1 < len(point):

    # mendefinisikan koordinat x dan y
    x1 = point[i]
    x2 = point[i + 1]

    y1 = data.iloc[point[i], 3]
    y2 = data.iloc[point[i + 1], 3]

    # menghitung kemiringan
    kemiringan = predict_slope(data.iloc[x1:x2, :])

    # Menambahkan garis
    fig.add_shape(type="line",
                  x0=data.index[x1],
                  y0=y1,
                  x1=data.index[x2],
                  y1=y2,
                  line=dict(color="red" if kemiringan < 0 else "green", width=3) )
    
    # menemukan nilai tengah
    X_tengah = int((x1 + x2) / 2)
    Y_tengah = int(data.iloc[x1, 3] + data.iloc[x2, 3]) / 2

    # Menambahkan teks
    fig.add_annotation(x=data.index[X_tengah],
                      y=Y_tengah,
                      text=f"{kemiringan:.3f}",
                      showarrow=False,
                      arrowhead=1,
                      bgcolor="#363636",
                      bordercolor="#666666"
                      )

# Update layout plot
fig.update_layout(title="Visualisasi Trend menggunakan metode kemiringan",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  height=800)

st.plotly_chart(fig)