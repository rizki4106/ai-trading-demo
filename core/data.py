import mplfinance as mf
import numpy as np
from PIL import Image
import pandas as pd

# ml
from core.controller import  device_diagnostic
from core.model import initialize_model_v1

# device diagnostic
device = device_diagnostic()

# initialize trained model
model_path = "model/model-1.pth"
model = initialize_model_v1(device=device, saved_weight_path=model_path)
model.eval()


def create_image(data : list = []):
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