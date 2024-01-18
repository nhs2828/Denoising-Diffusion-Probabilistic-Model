from typing import Optional
from functools import partial
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class MnistDataset(Dataset):
    def __init__(self, ds):
        self.images = torch.from_numpy(ds.images.data()).unsqueeze(1) / 255.0
        self.labels = ds.labels.data()
        self.channels = 1
        print(self.images[0, 0].shape)
        self.image_shape = self.images[0, 0].shape

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
    
def show_images(images, nrow=8):
    """Show images

    Args:
        images (torch.Tensor): The batch of images
        nrow (int, optional): The number of images per row. Defaults to 8.
    """
    images_3c = images.repeat(1, 3, 1, 1) if images.shape[1] != 3 else images
    images_3c = images_3c.double().clamp(0, 1)
    grid = make_grid(images_3c, nrow=nrow).permute((1, 2, 0))
    plt.imshow(grid.cpu())
    plt.show()

class Schedule:
    def __init__(self, betas):
        # Store basic information
        self.timesteps = len(betas)
        self.betas = betas
        self.alphas = 1.0 - betas
        
        # Pre-compute useful values:
        # use them in your code!
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod_prev = nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


class LinearSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        super().__init__(torch.linspace(beta_start, beta_end, timesteps))


class QuadraticSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        super().__init__(
            torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        )


class SigmoidSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        super().__init__(torch.sigmoid(betas) * (beta_end - beta_start) + beta_start)

def temporal_gather(a: torch.Tensor, t: torch.LongTensor, x_shape):
    """Gather values from tensor `a` using indices from `t`.

    Adds dimensions at the end of the tensor to match with the number of dimensions
    of `x_shape`
    """
    batch_size = len(t)
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)