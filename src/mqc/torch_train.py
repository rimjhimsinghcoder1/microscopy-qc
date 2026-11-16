"""Training loop for the PyTorch models, written out step by step.

The loop is kept explicit (no high-level trainer library) so every step is
visible: shuffle, forward pass, loss, backward pass, optimiser step, and
early stopping on the validation loss.
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from mqc.cnn import FusionNet, SmallCNN

TORCH_MODELS = {"cnn", "fusion"}


@dataclass
class TorchClassifier:
    """A trained network plus what is needed to use it on new data."""

    net: nn.Module
    feature_mean: np.ndarray
    feature_std: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)

    def _tensors(self, images: np.ndarray, features: np.ndarray):
        x = torch.from_numpy(images.astype(np.float32))
        f = (features - self.feature_mean) / self.feature_std
        return x, torch.from_numpy(f.astype(np.float32))

    @torch.no_grad()
    def predict_proba(self, images: np.ndarray, features: np.ndarray) -> np.ndarray:
        self.net.eval()  # BatchNorm uses its stored statistics, not the batch's
        x, f = self._tensors(images, features)
        return torch.softmax(self.net(x, f), dim=1).numpy()

    def predict(self, images: np.ndarray, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(images, features).argmax(axis=1)


def build_net(name: str, n_classes: int, n_features: int, width: int) -> nn.Module:
    if name == "cnn":
        return SmallCNN(n_classes, width)
    if name == "fusion":
        return FusionNet(n_classes, n_features, width)
    raise ValueError(f"unknown torch model {name!r}")


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train_torch_model(
    name: str,
    params: dict[str, Any],
    data: dict[str, np.ndarray],
    features: np.ndarray,
    n_classes: int,
    seed: int,
) -> TorchClassifier:
    """Train on `split_train`, keep the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train, val = data["split_train"], data["split_val"]
    # Feature scaling statistics come from the training split only, so no
    # information from validation or test leaks into training.
    mean = features[train].mean(axis=0)
    std = features[train].std(axis=0) + 1e-8

    net = build_net(name, n_classes, features.shape[1], params.get("width", 16))
    model = TorchClassifier(net, mean, std)
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
    )
    loss_fn = nn.CrossEntropyLoss()

    x_train, f_train = model._tensors(data["images"][train], features[train])
    y_train = torch.from_numpy(data["labels"][train]).long()
    x_val, f_val = model._tensors(data["images"][val], features[val])
    y_val = torch.from_numpy(data["labels"][val]).long()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    patience = params.get("patience", 8)
    for epoch in range(params.get("epochs", 60)):
        net.train()
        total = 0.0
        for idx in _batches(len(train), params.get("batch_size", 64), rng):
            optimiser.zero_grad()  # 1. clear old gradients
            logits = net(x_train[idx], f_train[idx])  # 2. forward pass
            loss = loss_fn(logits, y_train[idx])  # 3. how wrong are we?
            loss.backward()  # 4. gradients of the loss
            optimiser.step()  # 5. update the weights
            total += loss.item() * len(idx)

        net.eval()
        with torch.no_grad():
            val_logits = net(x_val, f_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        model.history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Early stopping: stop when validation loss has not improved for
        # `patience` epochs, and restore the best weights seen.
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = (
                val_loss,
                copy.deepcopy(net.state_dict()),
                0,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    net.load_state_dict(best_state)
    return model
