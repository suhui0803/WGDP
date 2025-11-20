import torch
import torch.nn.functional as F

def rmse(output, target, reduction="mean"):
    with torch.no_grad():
        scalar = F.mse_loss(output, target, reduction=reduction)
        scalar = torch.sqrt(scalar)
    return scalar
   
