import torch
from torchvision import transforms
import numpy as np
from skimage.transform import resize


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ct_slice, seg_mask = sample["ct_slice"], sample["seg_mask"]
        ct_slice = torch.from_numpy(ct_slice).permute(2, 0, 1)
        seg_mask = torch.from_numpy(seg_mask)
        return {"ct_slice": ct_slice, "seg_mask": seg_mask}


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        ct_slice, seg_mask = sample["ct_slice"], sample["seg_mask"]
        h, w = ct_slice.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        ct_slice = resize(ct_slice, (new_h, new_w), anti_aliasing=True)
        # nearest-neighbor interpolation for masks
        seg_mask = seg_mask.astype(np.bool)
        seg_mask = resize(seg_mask, (new_h, new_w)).astype(np.int64)

        return {"ct_slice": ct_slice, "seg_mask": seg_mask}


class Clip:
    """Clip pixel values"""
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, sample):
        ct_slice, seg_mask = sample["ct_slice"], sample["seg_mask"]
        ct_slice = np.clip(ct_slice, self.low, self.high)
        return {"ct_slice": ct_slice, "seg_mask": seg_mask}


class GlobalStandardize:
    """Convert pixels value range -> (-1, 1)"""
    def __call__(self, sample):
        ct_slice, seg_mask = sample["ct_slice"], sample["seg_mask"]

        mu, sigma = ct_slice.mean(), ct_slice.std()
        ct_slice = (ct_slice - mu) / sigma

        return {"ct_slice": ct_slice, "seg_mask": seg_mask}


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        ct_slice, seg_mask = sample["ct_slice"], sample["seg_mask"]
        ct_slice = transforms.Normalize(self.mean, self.std)(ct_slice)

        return {"ct_slice": ct_slice, "seg_mask": seg_mask}


DEFAULT_TRANSFORM = transforms.Compose([
    Rescale(256),
    Clip(-1000, 1000),
    ToTensor(),
    Normalize(mean=[0.], std=[1000.]),
])
