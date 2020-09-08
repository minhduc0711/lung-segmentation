from functools import lru_cache
import glob
import os
from pathlib import Path
from typing import List, Union
import time

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from tqdm import tqdm

from torch.utils.data import Dataset
from .utils import get_common_ids


class PlethoraDataset(Dataset):
    """
    Dataset URL: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327"
    """
    def __init__(
        self,
        ct_dir=None,
        mask_dir=None,
        metadata_path=None,
        ct_ids=None,
        transform=None,
    ):
        if metadata_path:
            self.metadata = self.load_metadata(metadata_path, ct_ids)
        else:
            assert (
                ct_dir is not None and mask_dir is not None
            ), "You have to provide either metadata_path or ct_dir & mask_dir"
            if ct_ids is None:
                # only include CT scans with corresponding masks available
                ct_ids = get_common_ids(ct_dir, mask_dir)
            self.generate_metadata(ct_dir, mask_dir, ct_ids)
        self.transform = transform

    def load_metadata(self, metadata_path, ct_ids):
        df = pd.read_csv(metadata_path)
        # only include specified CT scans
        if ct_ids is not None:
            df = df.loc[df["ct_id"].isin(ct_ids)]
        return df

    def generate_metadata(self, ct_dir, mask_dir, ct_ids):
        rows = []  # build a DataFrame from a list of dicts is omega faster

        for ct_id in tqdm(sorted(ct_ids), desc="Caching CT scans metadata"):
            dicom_paths = glob.glob(f"{ct_dir}/{ct_id}/*/*/*/*.dcm")
            dicom_paths = sorted(
                dicom_paths, key=lambda p: pydicom.read_file(p).SliceLocation
            )
            num_slices = len(dicom_paths)
            mask_path_str = "{}/{}/{}.npy"
            # store a dict of:
            # sample_idx -> (CT slice idx, CT slice dicom path, seg mask path)
            for slice_idx in range(num_slices):
                img_path = os.path.realpath(dicom_paths[slice_idx])
                mask_path = mask_path_str.format(mask_dir, ct_id, slice_idx)
                mask_path = os.path.realpath(mask_path)

                rows.append({"ct_id": ct_id,
                             "img_path": img_path,
                             "mask_path": mask_path})
        # save metadata to disk
        cols = ["ct_id", "img_path", "mask_path"]
        self.metadata = pd.DataFrame(rows, columns=cols)

        dataset_name = os.path.basename(os.path.normpath(ct_dir))
        timestamp = int(time.time())
        save_path = f"data/processed/{dataset_name}_metadata_{timestamp}.csv"
        self.metadata.to_csv(save_path, index=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path, mask_path = row["img_path"], row["mask_path"]

        dicom_file = pydicom.read_file(img_path)
        # convert to HU scale
        img = apply_modality_lut(dicom_file.pixel_array, dicom_file)
        img = img.astype(np.float32)
        mask = np.load(mask_path)
        sample = {"img": img, "mask": mask}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class KmaderDataset(Dataset):
    """
    Dataset URL: https://www.kaggle.com/kmader/finding-lungs-in-ct-data.
    Used in BCDU-net paper.
    """
    def __init__(self,
                 raw_path: str,
                 ct_ids: List[str] = None,
                 transform=None):
        def extract_id(p):
            return p.name.split(".")[0][-4:]

        raw_path = Path(raw_path)
        img_paths = list(raw_path.glob("IMG*"))
        if ct_ids is not None:
            img_paths = [p for p in img_paths
                         if extract_id(p) in ct_ids]
        img_paths.sort(key=extract_id)

        imgs = []
        masks = []
        for img_path in tqdm(img_paths, desc="Loading CT scans"):
            mask_path = str(img_path).replace("IMG", "MASK")
            img_file = nib.load(img_path)
            mask_file = nib.load(mask_path)

            img_arr = np.array(img_file.dataobj, dtype=np.float32)
            mask_arr = np.array(mask_file.dataobj, dtype=np.int64)
            # convert masks to values in {0, 1}
            mask_arr[mask_arr > 0] = 1

            imgs.append(img_arr)
            masks.append(mask_arr)

        assert len(imgs) != 0, f"No data were found in {str(raw_path)}"
        self.imgs = np.concatenate(imgs)
        self.masks = np.concatenate(masks)
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img, mask = self.imgs[idx], self.masks[idx]

        sample = {"img": img, "mask": mask}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class Covid19Dataset(Dataset):
    """
    Dataset URL: https://zenodo.org/record/3757476#.Xpz8OcgzZPY

    This dataset is only used for testing purposes at the moment,
    so random sample access is super slow.
    """
    def __init__(self,
                 ct_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 ct_ids=None,
                 transform=None):
        avail_ct_ids = set(os.listdir(ct_dir))
        avail_mask_ids = set(os.listdir(mask_dir))
        diff_ids = (avail_ct_ids - avail_mask_ids).union(
            avail_mask_ids - avail_mask_ids)
        assert len(diff_ids) == 0, \
            f"Found difference in CT and mask dirs: {diff_ids}"

        ct_dir = Path(ct_dir)
        mask_dir = Path(mask_dir)

        rows = []
        for mask_path in sorted(mask_dir.iterdir()):
            ct_id = mask_path.stem.split(".")[0]
            if ct_ids is not None and ct_id not in ct_ids:
                continue
            mask_file = nib.load(mask_path)
            num_slices = mask_file.dataobj.shape[-1]
            for i in range(num_slices):
                rows.append({"slice_idx": i,
                             "ct_id": ct_id,
                             "img_path": str(ct_dir / mask_path.name),
                             "mask_path": str(mask_path)})
        self.metadata = pd.DataFrame(rows, columns=list(rows[0].keys()))
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        slice_idx, img_path, mask_path = \
            row["slice_idx"], row["img_path"], row["mask_path"]
        imgs = self.load_nifti_arr(img_path, dtype=np.float32)
        masks = self.load_nifti_arr(mask_path, dtype=np.int64,
                                    is_mask=True)

        sample = {"img": imgs[slice_idx], "mask": masks[slice_idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    @lru_cache(maxsize=4)
    def load_nifti_arr(self, path: Union[str, Path],
                       dtype=np.float32,
                       is_mask: bool = False):
        nifti_file = nib.load(path)
        arr = np.array(nifti_file.dataobj)
        arr = np.rot90(arr, k=1)
        arr = arr.transpose(2, 0, 1)
        if is_mask:
            arr[arr > 0] = 1
        arr = np.ascontiguousarray(arr, dtype=dtype)
        return arr
