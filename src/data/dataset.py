import glob
import os

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from tqdm import tqdm

from torch.utils.data import Dataset


class NSCLCDataset(Dataset):
    def __init__(
        self,
        ct_dir=None,
        mask_dir=None,
        metadata_path=None,
        ct_ids=None,
        hu_range=(-1000, 1000),
        transform=None,
    ):
        if metadata_path:
            self.metadata = pd.read_csv(metadata_path)
        else:
            assert (
                ct_dir and mask_dir
            ), "You have to provide either metadata_path or ct_dir & mask_dir"
            if not ct_ids:
                # only include CT scans with corresponding masks available
                img_ids = set(os.listdir(ct_dir))
                seg_ids = set(os.listdir(mask_dir))
                ct_ids = list(img_ids.intersection(seg_ids))

            self.load_metadata(ct_dir, mask_dir, ct_ids)
        self.hu_range = hu_range
        self.transform = transform

    def load_metadata(self, ct_dir, mask_dir, ct_ids):
        cumu_num_slices = 0
        rows = []  # build a DataFrame from a list of dicts is omega faster

        for ct_id in tqdm(ct_ids, desc="Caching CT scans metadata"):
            dicom_paths = glob.glob(f"{ct_dir}/{ct_id}/*/*/*/*.dcm")
            dicom_paths = sorted(
                dicom_paths, key=lambda p: pydicom.read_file(p).SliceLocation
            )
            num_slices = len(dicom_paths)
            mask_path_str = "{}/{}/{}.npy"
            # store a dict of:
            # sample_idx -> (CT slice idx, CT slice dicom path, seg mask path)
            for slice_idx in range(num_slices):
                img_path = os.path.abspath(dicom_paths[slice_idx])
                mask_path = mask_path_str.format(mask_dir, ct_id, slice_idx)
                mask_path = os.path.abspath(mask_path)

                rows.append({"img_path": img_path, "mask_path": mask_path})
            cumu_num_slices += num_slices

        # save metadata dict to disk
        dataset_name = os.path.basename(os.path.normpath(ct_dir))
        self.metadata = pd.DataFrame(rows, columns=["img_path", "mask_path"])
        self.metadata.to_csv(f"data/processed/{dataset_name}_metadata.csv",
                             index=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path = self.metadata.iloc[idx]

        dicom_file = pydicom.read_file(img_path)
        # convert to HU scale
        img = apply_modality_lut(dicom_file.pixel_array, dicom_file)
        img = img.astype(np.float32)
        # add explicit channel dim to images
        img = np.expand_dims(img, axis=-1)
        mask = np.load(mask_path)

        sample = {"img": img, "mask": mask}
        if self.transform:
            sample = self.transform(sample)

        return sample
