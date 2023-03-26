from torchvision import transforms
import torch

def device_diagnostic():
  """Check the available device
  
  returns:
  device : str -> cuda | cpu
  """
  return "cuda" if torch.cuda.is_available() else "cpu"