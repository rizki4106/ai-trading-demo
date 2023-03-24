from torchvision import models
import torch

def initialize_model_v1(device : str, saved_weight_path : str):
    """Initialize model v1 with 2 data training (up and down)
    
    Args:
        device : str -> cuda if cuda is available and cpu if is'nt
    
    Returns:
        model : pytorch trained model
    """

    # get trained weight from efficientnet-b0
    weights = models.EfficientNet_B0_Weights.DEFAULT

    # initialize model
    model = models.efficientnet_b0(weights=weights).to(device)

    # freezing model weights
    for param in model.features.parameters():
        param.requires_grad = False

    # create classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=2,
                        bias=True)
    ).to(device)

    # load saved state dict
    model.load_state_dict(torch.load(saved_weight_path, map_location=torch.device('cpu')))
    model.eval()

    return model