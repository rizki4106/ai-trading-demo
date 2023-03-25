import mplfinance as mf
import numpy as np
from PIL import Image
import pandas as pd

class DataCreation:
  """
  This class used for create data needs
  """
  def create_image(self, data : list = []):
    """Create candle stick or chart pattern image and return it as PIL Image class.
    Args:
        data : pandas data frame -> list historycal data in pandas data frame format from yfinance or other
    Returns:
        image : PIL Image -> PIL image class
    """

    # make candlestick style just as the same with trading view color
    mc = mf.make_marketcolors(up='#26A69A', edge='inherit', down='#EF5350', wick={"up" : '#26A69A', 'down': '#EF5350'})

    # configuring figure style
    s  = mf.make_mpf_style(gridstyle="", marketcolors=mc, edgecolor="#ffffff")

    # create figure
    fig = mf.figure(style=s)
    ax = fig.add_subplot(1, 1, 1)

    # remove x and y label
    ax.axis(False)

    # create candle stick
    mf.plot(data,ax=ax,type="candle")

    # draw candle stick into canvas
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    candle_arr = np.array(fig.canvas.renderer.buffer_rgba())

    # convert numpy array to PIL image
    img = Image.fromarray(candle_arr).convert("RGB")

    return img
  
data_creation = DataCreation()

def visualize_result(data : pd.DataFrame, n_candle: int, class_name : dict):
    
    # kelompokan data menjadi panjang data / n buah candle
    n = int(len(data) / n_candle)

    # bagi baris menjadi kelompok
    kelompok = pd.cut(data.index, bins=n, labels=False)

    # kelompokkan data berdasarkan kelompok yang telah dibuat
    kelompok_data = data.groupby(kelompok)
    
    # ubah data [open, high, low, close, volume] menjadi gambar
    preds = []

    for i, (name, group) in enumerate(kelompok_data):

        # generate image
        img = data_creation.create_image(group.iloc[:, :-2])

        # lakukan prediksi disini
        preds.append(1)
        
    # menentukan titik candle stick untuk ditampilkan buy atau sell
    points = np.arange(n_candle, int(len(data)), n_candle)
    
    # buat titik anotasi untuk menulis buy or sell
    annts = []

    for pr, po in zip(preds, points):

        ann = dict(x=data.index[po],
                   y=data['High'][po],
                   xref='x',
                   yref='y',
                   showarrow=True,
                   arrowhead=7,
                   ax=0,
                   ay=-40 if pr == 1 else 40,
                   text=class_name[pr])

        annts.append(ann)
    
    # return titik anotasi
    return annts