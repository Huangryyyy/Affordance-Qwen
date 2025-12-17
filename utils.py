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


def expand2square(img):
    width, height = img.size
    if width == height:
        return img
    new_side = max(width, height)
    if img.mode == "RGB":
        square_canvas = Image.new("RGB", (new_side, new_side), (0, 0, 0))
    elif img.mode == "L":
        square_canvas = Image.new("L", (new_side, new_side), 0)
    else:
        print("Unknown image mode")
    x_offset = (new_side - width) // 2
    y_offset = (new_side - height) // 2
    square_canvas.paste(img, (x_offset, y_offset))
    return square_canvas


import torch.nn as nn
import torch.nn.functional as F


class SegmentationFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * bce_loss
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def save_example(pred, gt, origin_img, file_path, gap=0.1):
    img1 = pred.float().cpu().detach().numpy()
    img2 = gt.float().cpu().detach().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(img1, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("pred", fontsize=14)
    ax1.axis("off")

    ax2.imshow(img2, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("gt", fontsize=14)
    ax2.axis("off")

    ax3.imshow(origin_img)
    ax3.set_title("image", fontsize=14)
    ax3.axis("off")

    plt.subplots_adjust(wspace=gap)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches="tight", dpi=150)

    plt.close(fig)


def cal_kl(
    pred, gt, eps=1e-12
):  # Input: [B, H, W]  Output: [B]. The same for the two functions below
    pred_sum = pred.sum(dim=(-2, -1), keepdim=True)
    gt_sum = gt.sum(dim=(-2, -1), keepdim=True)
    map1 = pred / (pred_sum + eps)
    map2 = gt / (gt_sum + eps)
    term = map2 * torch.log(map2 / (map1 + eps) + eps)
    kld = torch.sum(term, dim=(-2, -1))
    return kld


def cal_sim(pred, gt, eps=1e-12):
    pred_sum = pred.sum(dim=(-2, -1), keepdim=True)
    gt_sum = gt.sum(dim=(-2, -1), keepdim=True)
    map1 = pred / (pred_sum + eps)
    map2 = gt / (gt_sum + eps)
    intersection = torch.minimum(map1, map2)
    return torch.sum(intersection, dim=(-2, -1))


def cal_nss(pred, gt, eps=1e-12):
    u = torch.mean(pred, dim=(-2, -1), keepdim=True)
    std = torch.std(pred, dim=(-2, -1), keepdim=True)
    smap = (pred - u) / (std + eps)
    gt_min = torch.amin(gt, dim=(-2, -1), keepdim=True)
    gt_max = torch.amax(gt, dim=(-2, -1), keepdim=True)
    fixation_map = (gt - gt_min) / (gt_max - gt_min + eps)
    threshold = 0.1
    fixation_map = (
        fixation_map > threshold
    ).float()  # we use a threshold here two get binary gt mask (if it wasn't binary before)
    nss_sum = torch.sum(smap * fixation_map, dim=(-2, -1))
    count = torch.sum(fixation_map, dim=(-2, -1))
    nss = nss_sum / (count + eps)
    return nss
