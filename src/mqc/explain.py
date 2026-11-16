"""Grad-CAM: which parts of the image drove the CNN's decision?

Selvaraju et al. (2017). For a chosen class, take the gradient of its score
with respect to a convolutional layer's output, average it per channel to get
one weight per channel, and sum the channels with those weights. Positive
values mark regions that pushed the score up.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mqc.cnn import FusionNet, SmallCNN


def _target_layer(net: nn.Module) -> nn.Module:
    """The 14x14 conv block: finer than the last (7x7) block on 28x28 images."""
    cnn = net.cnn if isinstance(net, FusionNet) else net
    if not isinstance(cnn, SmallCNN):
        raise TypeError("Grad-CAM needs a SmallCNN or FusionNet")
    return cnn.features[3]


def grad_cam(
    net: nn.Module,
    images: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor | None = None,
) -> np.ndarray:
    """Return heatmaps (N, H, W) in [0, 1], one per image, for class `target`."""
    store: dict[str, torch.Tensor] = {}
    layer = _target_layer(net)
    handle = layer.register_forward_hook(
        lambda module, inp, out: store.__setitem__("act", out)
    )
    try:
        net.eval()
        images = images.clone().requires_grad_(True)
        scores = net(images, features)
        selected = scores[torch.arange(len(target)), target].sum()
        grads = torch.autograd.grad(selected, store["act"])[0]
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)  # one weight per channel
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear")[:, 0]
    cam = cam.detach().numpy()
    peak = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
    return np.divide(cam, peak, out=np.zeros_like(cam), where=peak > 0)
