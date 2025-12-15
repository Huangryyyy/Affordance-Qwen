import random
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch.nn as nn
import torch.nn.functional as F


class SegmentationFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * bce_loss
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def save_two_tensors(tensor1, tensor2, file_path, title1="pred", title2="gt", gap=0.1):
    img1 = tensor1.float().cpu().detach().numpy()
    img2 = tensor2.float().cpu().detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img1, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(title1, fontsize=14)
    ax1.axis("off")

    ax2.imshow(img2, cmap="gray", vmin=0, vmax=1)
    ax2.set_title(title2, fontsize=14)
    ax2.axis("off")

    plt.subplots_adjust(wspace=gap)

    plt.savefig(file_path, bbox_inches="tight", dpi=150)

    plt.close(fig)
