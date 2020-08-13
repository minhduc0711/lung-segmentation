import torch
from torchvision import transforms
import numpy as np
from skimage.transform import resize


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return {"img": img, "mask": mask}


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
        img, mask = sample["img"], sample["mask"]
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(img, (new_h, new_w), anti_aliasing=True)
        # nearest-neighbor interpolation for masks
        mask = mask.astype(np.bool)
        mask = resize(mask, (new_h, new_w)).astype(np.int64)

        return {"img": img, "mask": mask}


class Clip:
    """Clip pixel values"""
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        img = np.clip(img, self.low, self.high)
        return {"img": img, "mask": mask}


class GlobalStandardize:
    """Convert pixels value range -> (-1, 1)"""
    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]

        mu, sigma = img.mean(), img.std()
        img = (img - mu) / sigma

        return {"img": img, "mask": mask}


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        img = transforms.Normalize(self.mean, self.std)(img)

        return {"img": img, "mask": mask}


DEFAULT_TRANSFORM = transforms.Compose([
    Rescale(256),
    Clip(-1000, 1000),
    ToTensor(),
    Normalize(mean=[0.], std=[1000.]),
])
