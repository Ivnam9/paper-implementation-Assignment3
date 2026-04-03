import torch
import torch.nn.functional as F

def photometric_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2))

def smoothness_loss(depth):
    dx = torch.mean(torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:]))
    dy = torch.mean(torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :]))
    return dx + dy