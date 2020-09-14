from typing import Tuple, Union

import torch
import numpy as np
from scipy.ndimage.morphology import binary_opening, binary_fill_holes
from skimage.transform import resize


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        sample["img"], sample["mask"] = img, mask
        return sample 


class Resize:
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self,
                 output_size: Union[int, Tuple[int, int]]):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        h, w = img.shape[-2:]
        if self.output_size == (h, w):
            return sample
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])
        img = resize(img, (new_h, new_w), anti_aliasing=True)
        # nearest-neighbor interpolation for masks
        mask = mask.astype(np.bool)
        mask = resize(mask, (new_h, new_w)).astype(np.int64)

        sample["img"], sample["mask"] = img, mask

        return sample


class Clip:
    """Clip pixel values"""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, sample):
        img = sample["img"]
        img = np.clip(img, self.low, self.high)

        sample["img"] = img
        return sample


class GlobalStandardize:
    """Convert pixels value range -> (-1, 1)"""

    def __call__(self, sample):
        img = sample["img"]

        mu, sigma = img.mean(), img.std()
        img = (img - mu) / sigma
        sample["img"] = img

        return sample


class Normalize:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, sample):
        img = sample["img"]

        # https://en.wikipedia.org/wiki/Normalization_(image_processing)
        norm_img = img - img.min()
        norm_img *= (self.high - self.low) / (img.max() - img.min())
        norm_img += self.low

        sample["img"] = norm_img

        return sample


class ExtractMaskAroundLungs:
    """Extract regions of the body that are not lungs, using lungs mask.
    NOTE: this transformation must be performed after HU clipping
    """

    def __call__(self, sample):
        img, mask_lungs = sample["img"].squeeze(), sample["mask"]

        mask_around_lungs = (img > img.min()).astype(np.int64)
        mask_around_lungs = binary_opening(mask_around_lungs,
                                           structure=np.ones((15, 15)))
        # post-processing
        mask_body = mask_around_lungs + mask_lungs
        mask_body[mask_body > 0] = 1
        mask_body = binary_fill_holes(mask_body, structure=np.ones((5, 5)))
        mask_body = mask_body.astype(np.int64)
        mask_around_lungs = mask_body - mask_lungs

        sample["mask"] = mask_around_lungs
        return sample


def extract_mask_lungs(mask_around_lungs):
    if isinstance(mask_around_lungs, torch.Tensor):
        mask_around_lungs = mask_around_lungs.numpy()

    mask_body = binary_fill_holes(mask_around_lungs).astype(np.int64)
    mask_lungs = mask_body - mask_around_lungs
    return mask_lungs


class ExtractMaskLungs:
    """Extract lungs mask from around-lungs masks."""

    def __call__(self, sample):
        mask_around_lungs = sample["mask"]
        sample["mask"] = extract_mask_lungs(mask_around_lungs)
        return sample
