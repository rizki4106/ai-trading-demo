from torchvision import models, transforms
from PIL import Image
import torch
import pandas as pd

class ConfirmModelV1:
    """
    Predict a confirmation trend to action
    """

    __model = None
    class_name = {}
    device = 'cpu'

    def __init__(self,
                 state_dict_path : str, 
                 device : str,
                 class_name : dict) -> None:
        """
        Args:
            state_dict_path : str -> saved state dict from training usualy .pth or pt file
            device : str -> available device cpu or cuda
        """

        # initialize the model
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.__model = models.efficientnet_b0(weights=weights).to(device)

        # initialize classifier
        self.__model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=2,
                            bias=True)
        ).to(device)

        if device == "cuda":
            self.__model.load_state_dict(torch.load(state_dict_path))
        else:
            self.__model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        self.__model.eval()

        # set the classname
        self.class_name = class_name

        # set the available device
        self.device = device

    def predict(self, image : Image):
        """Predict trend market up or down
        Args:
            image : PIL.Image -> chart pattern image
        Returns:
            result : dict -> probability, class_int, class_name
        """

        # define image transformation
        image_size = (355, 563)
        transformer = transforms.Compose([
            
            # Resize image to n image_size
            transforms.Resize(image_size),

            #turn image into tensor
            transforms.ToTensor()
        ])

        # run prediction
        with torch.inference_mode():
            
            # turn image into tensor
            image = transformer(image)

            # predict image
            preds = self.__model(image.unsqueeze(0).to(self.device)).squeeze()

            # fit the output to softmax activation function
            preds = torch.softmax(preds, dim=-1)

            # get id of the maximum value
            idx = preds.argmax().item()

            return {
                "probability": preds[idx],
                "class_int": idx,
                "class_name": self.class_name[idx]
            }

class CandleStick:

    # define rule name in dictionary
    rule_name = {
        0: "sell",
        1: "buy",
        2: "do nothing",
    }

    # define candle stick name in dictionary
    candlestick_name = {
        0: "bearish",
        1: "bullish",
    }

    def candlestick_type(self, df):
        """
        Given a candlestick dataframe return the type of candlestick
        
        @param df - dataframe with columns Close Open Low High High.
        """
        if df['Close'] < df['Open']:
            return 0
        elif df['Close'] > df['Open']:
            return 1
    
    def do_transaction(self, frame):
        """
        If 3 candle stick detected as bearish, bullish, bullish then buy
        If 3 candle stick detected as bullish, bearish, bearish then sell

        @param frame - dataframe with 3 rows and columns Close Open Low High High.
        @return dected_action -> 0 for sell, 1 for buy, 2 for do nothing
        """

        # define rule for buy
        # bearish, bulish, bulish
        buy_rule = [0,1,1]

        # define rule for sell
        # bulish, bearish, bearish
        sell_rule = [1,0,0]

        # detected rule buy or sell
        dected_rule = []

        # detect candle stick pattern from given 3 rows of dataframe
        for i in range(len(frame)):
            dected_rule.append(self.candlestick_type(frame.iloc[i, :]))

        # check if detected rule is equal to buy rule
        if dected_rule == buy_rule:

            # check if the last volume candle greater than second candle
            # this is must be solve
            # calculate the difference from the open and close prices

            difference_last = frame.iloc[-1, :]['Close'] - frame.iloc[-1, :]['Open']
            difference_last_2 = frame.iloc[-2, :]['Close'] - frame.iloc[-2, :]['Open']

            if difference_last > difference_last_2:
                return 1
            else:
                return 2

        # check if detected rule is equal to sell rule
        elif dected_rule == sell_rule:
            return 0
        else:
            return 2
        
    def test_candle_transaction(self, frame):
        """
        Given ohlc data in pandas dataframe
        then select 3 candle stick and do prediction

        @param: frame: pandas dataframe
        @return: list of annotation point and list of prediction result
        """

        n_candle = 3

        # define annotation point
        annt = []

        # define prediction result
        prediction = []

        # start sliding 3 sliding window to the right
        for i in range(len(frame)):

            # select 3 candle
            selected_candle = frame.iloc[i:n_candle, :]

            # if selected candle is 3 then do prediction
            if len(selected_candle) == 3:

                # do prediction
                preds = self.do_transaction(selected_candle)

                if preds == 1 or preds == 0:
                    annt.append(n_candle)
                    prediction.append(preds)

            # update n_candle by 1
            n_candle += 1

        return annt, prediction

class SignalConfirmV1:
    """
    Merge the ConvirmModelV1 as Signal prediction and Candlestick as a confirmation from the signal
    """

    def __init__(self, 
                 confirm_model : ConfirmModelV1, 
                 candlestick : CandleStick,
                 device : str) -> None:
        """
        Args:
            confirm_model : ConfirmModelV1 -> confirm model
            candlestick : CandleStick -> candle stick model
            device : str -> available device cpu or cuda
        """

        # initialize confirm model
        self.confirm_model = confirm_model

        # initialize candlestick model
        self.candlestick = candlestick

        # initialize device
        self.device = device

    def predict(self, image : Image, frame : pd.DataFrame):
        """
        Predict trend market up or down
        Args:
            image : PIL.Image -> chart pattern image
            frame : pd.DataFrame -> ohlc data
        Returns:
            result : dict -> probability, class_int, class_name
        """

        # predict signal
        signal = self.confirm_model.predict(image)

        # get the annotation point and prediction result
        confirmation = self.candlestick.do_transaction(frame)

        if signal['class_int'] == 1 and confirmation == 1:
            return {
                "probability": signal['probability'],
                "class_int": 1,
                "class_name": "buy"
            }
        elif signal['class_int'] == 0 and confirmation == 0:
            return {
                "probability": signal['probability'],
                "class_int": 0,
                "class_name": "sell"
            }
        else:
            return {
                "probability": -1,
                "class_int": 2,
                "class_name": "do nothing"
            }