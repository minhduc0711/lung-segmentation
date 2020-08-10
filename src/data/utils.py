import glob

import numpy as np

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import nibabel as nib


def read_ct_scan(ct_dir, hu_range=None):
    dicom_paths = glob.glob(f"{ct_dir}/*.dcm")
    dicom_files = sorted([pydicom.read_file(path) for path in dicom_paths],
                         key=lambda ds: ds.SliceLocation)
    ct_arr = [apply_modality_lut(ds.pixel_array, ds) for ds in dicom_files]
    ct_arr = np.array(ct_arr, dtype=np.float32)
    return ct_arr


def read_seg_masks(fpath):
    seg_file = nib.load(fpath)
    seg_masks = seg_file.dataobj[:].astype(np.float32)
    # move z dim to 1st dim
    seg_masks = seg_masks.transpose(2, 0, 1)
    # match mask to CT image orientation
    seg_masks = np.rot90(seg_masks, k=1, axes=(1, 2))
    # remove left, right lung information
    seg_masks[seg_masks > 0] = 1.0
    seg_masks = np.ascontiguousarray(seg_masks)
    return seg_masks

def read_seg_mask(fpath, slice_idx):
    seg_file = nib.load(fpath)
    seg_mask = seg_file.dataobj[..., slice_idx].astype(np.int64)
    # match mask to CT image orientation
    seg_mask = np.rot90(seg_mask, k=1)
    # remove left, right lung information
    seg_mask[seg_mask > 0] = 1
    seg_mask = np.ascontiguousarray(seg_mask)
    return seg_mask
