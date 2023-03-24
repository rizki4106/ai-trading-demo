from torchvision import transforms
import torch


def device_diagnostic():
  return "cuda" if torch.cuda.is_available() else "cpu"

def predict(model : None, image : torch.tensor, device : str):
  
  """Run prediction

  Args:
    model : torch model
    image : torch.tensor -> image that you want to predict
    device : str -> cuda if cuda is available and cpu if is'nt

  Returns:
    probability : float -> probability prediction
    class : int -> class data
  """

  image_size = (355, 563)

  transformer = transforms.Compose([
      
      # Resize image to n image_size
      transforms.Resize(image_size),

      #turn image into tensor
      transforms.ToTensor()
  ])

  with torch.inference_mode():

    # turn image into tensor
    image = transformer(image)

    # predict image
    preds = model(image.unsqueeze(0).to(device)).squeeze()

    # fit the output to softmax activation function
    preds = torch.softmax(preds, dim=-1)

    # get id of the maximum value
    idx = preds.argmax().item()

    return preds[idx], idx