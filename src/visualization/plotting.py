import numpy as np
import matplotlib.pyplot as plt


def plot_samples(
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

    for ax, ct_slice, gt_mask in zip(
        axes.flatten(), X_batch[:num_shown], y_batch[:num_shown]
    ):
        ct_slice = ct_slice.squeeze()
        ct_slice -= ct_slice.min()  # shift to all positive values
        ct_slice /= ct_slice.max()  # normalize

        color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        color_mask[gt_mask == 1.0] = mask_color

        ax.axis("off")
        ax.imshow(ct_slice, cmap="gray")
        ax.imshow(color_mask, alpha=mask_alpha)
