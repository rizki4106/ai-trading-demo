from torchvision import models, transforms
from PIL import Image
import torch

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