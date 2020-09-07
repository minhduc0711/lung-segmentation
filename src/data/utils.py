import os
import glob

import numpy as np
import torch

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import nibabel as nib


def read_ct_scan(ct_dir, hu_range=None):
    dicom_paths = glob.glob(f"{ct_dir}/*.dcm")
    dicom_files = sorted(
        [pydicom.read_file(path) for path in dicom_paths],
        key=lambda ds: ds.SliceLocation,
    )
    ct_arr = [apply_modality_lut(ds.pixel_array, ds) for ds in dicom_files]
    ct_arr = np.array(ct_arr, dtype=np.float32)
    return ct_arr


def read_masks(fpath):
    seg_file = nib.load(fpath)
    masks = seg_file.dataobj[:].astype(np.float32)
    # move z dim to 1st dim
    masks = masks.transpose(2, 0, 1)
    # match mask to CT image orientation
    masks = np.rot90(masks, k=1, axes=(1, 2))
    # remove left, right lung information
    masks[masks > 0] = 1.0
    masks = np.ascontiguousarray(masks)
    return masks


def read_mask(fpath, slice_idx):
    seg_file = nib.load(fpath)
    mask = seg_file.dataobj[..., slice_idx].astype(np.int64)
    # match mask to CT image orientation
    mask = np.rot90(mask, k=1)
    # remove left, right lung information
    mask[mask > 0] = 1
    mask = np.ascontiguousarray(mask)
    return mask


def get_common_ids(*dirs):
    res = set.intersection(*[set(os.listdir(d)) for d in dirs])
    return sorted(list(res))
