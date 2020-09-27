import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize(img):
    """Normalize a ndarray image"""
    res = img.copy()
    res -= res.min()  # shift to range (0, inf)
    res /= res.max()  # normalize
    return res


def plot_batch(
    X_batch,
    y_batch,
    nrows,
    ncols,
    figsize=(15, 15),
    mask_color=(255, 0, 0),
    mask_alpha=0.15,
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    fig.patch.set_facecolor((1, 1, 1))

    H, W = y_batch.shape[-2:]
    for i, ax in enumerate(axes):
        img = X_batch[i]
        mask = y_batch[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = img.copy().squeeze()
        img -= img.min()  # shift to range (0, inf)
        img /= img.max()  # normalize

        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color_mask[mask == 1.0] = mask_color

        ax.axis("off")
        ax.imshow(img, cmap="gray")
        ax.imshow(color_mask, alpha=mask_alpha)


def plot_true_vs_pred(X, y_true, y_pred, figsize=None, mask_alpha=0.15,
                      subplot_labels=None):
    H, W = y_true.shape[-2:]

    batch_size = X.shape[0]
    if not figsize:
        figsize = (10, 5 * batch_size)
    else:
        figsize = (figsize[0], figsize[1] * batch_size)
    fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=figsize)
    fig.patch.set_facecolor((1, 1, 1))
    axes = axes.reshape(-1, 2)

    X = X.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    for i, ax_pair in enumerate(axes):
        img = normalize(X[i].squeeze())
        ax0, ax1 = ax_pair
        if i == 0:
            ax0.set_title("ground-truth")
            ax1.set_title("predicted")

        gt_mask = np.zeros((H, W, 3), dtype=np.uint8)
        gt_mask[y_true[i] == 1] = (0, 255, 0)
        ax0.axis("off")
        ax0.imshow(img, cmap="gray")
        ax0.imshow(gt_mask, alpha=mask_alpha)

        pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
        pred_mask[y_pred[i] == 1] = (255, 0, 0)
        ax1.axis("off")
        ax1.imshow(img, cmap="gray")
        ax1.imshow(pred_mask, alpha=mask_alpha)
        if subplot_labels is not None:
            ax1.set_ylabel(subplot_labels[i], rotation='horizontal',
                           fontweight="bold")
