from core.data import create_image
import pandas as pd

class TraderTester:
    """This class used for testing the ML Model for predicting
    the movement of the price like stock, crypto and etc

    Args:
        historycal_data : pandas DataFrame -> historical price data [open, low, high, close, volume]
        n_candle : int -> number of candle that will take for one prediction
        model : torch saved model -> pytorch saved model
        class_name : dict -> class name in integer and string
    """

    def __init__(self,
                 historical_data : pd.DataFrame, 
                 n_candle : int,
                 model : None,
                 ) -> None:
        
        # set historycal data to global properties
        self.data = historical_data

        # set number of candle that will take for one prediction
        self.n_candle = n_candle

        # 
        self.model = model

    def back_test(self):
        """Do the test using past data
        
        Returns:
            annotation : list -> annotation point of the transaction
        """
        # define start point where we should start taking the data
        start_point = 0

        # define numbers of candle stick for one iteration
        n_candle = self.n_candle

        # define first n_candle of candle stick
        candle = self.data.iloc[start_point:n_candle, :-2]

        # define length of the data
        data_length = len(self.data)

        # define the prediction
        history = ['sell']

        # define annotation
        annt = []

        # move every n_candle to the right
        for i in range(12, data_length):
            
            # update n_candle
            n_candle = i
            
            # update group of candle
            candle = self.data.iloc[start_point:n_candle, :-2]
            
            # turn from number to image
            img = create_image(candle)
            
            # run prediction
            preds = self.model.predict(img)

            # set prediction threshold
            treshold = 0.6

            # if the last transaction is sell then next transaction should buy
            if history[-1] == "sell" and preds['class_name'] == "buy":

                if preds['probability'] > treshold:

                    annt.append(
                        dict(x=self.data.index[i],
                            y=self.data['High'][i],
                            xref='x',
                            yref='y',
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-40 if preds['class_int'] == 1 else 40,
                            text=f'{preds["class_name"]}')
                    )

                    history.append("buy")

            # if the last transaction is buy then next transaction should sell
            elif history[-1] == "buy" and preds['class_name'] == "sell":

                if preds['probability'] > treshold:
                    annt.append(
                        dict(x=self.data.index[i],
                            y=self.data['High'][i],
                            xref='x',
                            yref='y',
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-40 if preds['class_int'] == 1 else 40,
                            text=f'{preds["class_name"]}')
                    )

                    history.append("sell")
            # update start point
            start_point += 1
        
        return annt