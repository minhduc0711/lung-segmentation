import os
from typing import Union, Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from .datasets import PlethoraDataset, KmaderDataset, Covid19Dataset
from .preprocess import Clip, ToTensor, Normalize, Resize, \
    ExtractMaskAroundLungs
from .utils import get_common_ids


class BaseImageSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        img_size: Union[int, Tuple[int, int]] = 512,
        clip_low: float = None,
        clip_high: float = None,
        invert_lungs: bool = False,
        pin_memory: bool = True,
        num_workers: int = 4,
        transform=None,
    ):
        super(BaseImageSegmentationDataModule, self).__init__()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        if transform is None:
            # default transform
            transform_list = [
                Resize(img_size),
                Clip(clip_low, clip_high),
                ToTensor(),
                transforms.Lambda(lambda sample: dict(sample,
                                                      img=sample["img"].unsqueeze(0))),
                Normalize(low=0, high=1),
            ]
            if invert_lungs:
                # this has to be placed after clipping
                transform_list.insert(2, ExtractMaskAroundLungs())
            self.dims = (
                (1, img_size, img_size)
                if isinstance(img_size, int)
                else (1, img_size[0], img_size[1])
            )
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class PlethoraDataModule(BaseImageSegmentationDataModule):
    RAW_CT_SCAN_DIR = "data/raw/NSCLC-Radiomics/"
    RAW_MASK_DIR = "data/raw/Thoracic_Cavities/"
    PROCESSED_MASK_DIR = "data/processed/plethora_masks/"

    METADATA_PATH = "data/processed/plethora_metadata.csv"

    def __init__(self, *args, **kwargs):
        super(PlethoraDataModule, self).__init__(*args, **kwargs)
        self.num_classes = 2  # lung vs no-lung

    def prepare_data(self):
        """Split NifTI masks into separate files containing 1 slice each"""
        # TODO: perform dir exists check and relocate code from split_NSCLC_masks.py
        pass

    def setup(self, stage: Optional[str] = None):
        train_ratio = 0.7
        val_ratio = 0.2

        ct_ids = get_common_ids(self.RAW_CT_SCAN_DIR, self.PROCESSED_MASK_DIR)
        num_train_scans = int(len(ct_ids) * train_ratio)
        num_val_scans = int(len(ct_ids) * val_ratio)
        train_scans = ct_ids[:num_train_scans]
        val_scans = ct_ids[num_train_scans:num_train_scans + num_val_scans]
        test_scans = ct_ids[num_train_scans + num_val_scans:]

        if stage == "fit" or stage is None:
            self.train_ds = PlethoraDataset(
                metadata_path=self.METADATA_PATH,
                transform=self.transform,
                ct_ids=train_scans,
            )
            self.val_ds = PlethoraDataset(
                metadata_path=self.METADATA_PATH,
                transform=self.transform,
                ct_ids=val_scans,
            )
        if stage == "test" or stage is None:
            self.test_ds = PlethoraDataset(
                metadata_path=self.METADATA_PATH,
                transform=self.transform,
                ct_ids=test_scans,
            )


class KmaderDataModule(BaseImageSegmentationDataModule):
    def __init__(self, *args, **kwargs):
        super(KmaderDataModule, self).__init__(*args, **kwargs)
        self.num_classes = 2  # lung vs no-lung

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            ds = KmaderDataset(
                raw_path="data/raw/kmader/3d_images/",
                ct_ids=["0002", "0031", "0078"],
                transform=self.transform,
            )
            train_ratio = 0.8
            num_train = int(train_ratio * len(ds))
            num_val = len(ds) - num_train

            self.train_ds, self.val_ds = random_split(
                dataset=ds,
                lengths=[num_train, num_val],
                generator=torch.Generator().manual_seed(25)
            )
        if stage == "test" or stage is None:
            self.test_ds = KmaderDataset(
                raw_path="data/raw/kmader/3d_images/",
                ct_ids=["0059"],
                transform=self.transform
            )


class Covid19DataModule(BaseImageSegmentationDataModule):
    RAW_CT_SCAN_DIR = "data/raw/COVID-19-CT-Seg_20cases/ct_scans/"
    RAW_MASK_DIR = "data/raw/COVID-19-CT-Seg_20cases/lung_masks/"

    def __init__(self, *args, **kwargs):
        super(Covid19DataModule, self).__init__(*args, **kwargs)
        self.num_classes = 2

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            raise NotImplementedError("Train/val is not supported yet")
        if stage == "test" or stage is None:

            # ignore pre-normalized ct scans
            ct_ids = os.listdir(self.RAW_CT_SCAN_DIR)
            ct_ids = sorted([os.path.basename(ct_id).split(".")[0] for ct_id in ct_ids])
            ct_ids = [ct_id for ct_id in ct_ids if ct_id.find("radio") == -1]
            self.test_ds = Covid19Dataset(ct_dir=self.RAW_CT_SCAN_DIR,
                                          mask_dir=self.RAW_MASK_DIR,
                                          transform=self.transform,
                                          ct_ids=ct_ids)

    def train_dataloader(self):
        raise NotImplementedError("Train/val is not supported yet")

    def val_dataloader(self):
        raise NotImplementedError("Train/val is not supported yet")
