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
    num_shown = nrows * ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    H, W = X_batch.shape[2:]
    for ax, ct_slice, gt_mask in zip(
        axes.flatten(), X_batch[:num_shown], y_batch[:num_shown]
    ):
        ct_slice = ct_slice.numpy().copy().squeeze()
        ct_slice -= ct_slice.min()  # shift to range (0, inf)
        ct_slice /= ct_slice.max()  # normalize

        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color_mask[gt_mask == 1.0] = mask_color

        ax.axis("off")
        ax.imshow(ct_slice, cmap="gray")
        ax.imshow(color_mask, alpha=mask_alpha)


def plot_true_vs_pred(X, y_true, y_pred, figsize=None, mask_alpha=0.15):
    H, W = X.shape[2:]
    batch_size = X.shape[0]
    if not figsize:
        figsize = (10, 5 * batch_size)
    fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=figsize)
    for i, ax_pair in enumerate(axes):
        ct_slice = normalize(X[i].numpy().squeeze())
        ax0, ax1 = ax_pair
        if i == 0:
            ax0.set_title("ground-truth")
            ax1.set_title("predicted")

        gt_mask = np.zeros((H, W, 3), dtype=np.uint8)
        gt_mask[y_true[i] == 1] = (0, 255, 0)
        ax0.axis("off")
        ax0.imshow(ct_slice, cmap="gray")
        ax0.imshow(gt_mask, alpha=mask_alpha)

        pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
        pred_mask[y_pred[i] == 1] = (255, 0, 0)
        ax1.axis("off")
        ax1.imshow(ct_slice, cmap="gray")
        ax1.imshow(pred_mask, alpha=mask_alpha)
