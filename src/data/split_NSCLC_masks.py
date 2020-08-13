""" This script does some preprocessing on segmentation masks and
then store each slice in a numpy file to speed up random data access
"""

import argparse as ap
import glob
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import tqdm

parser = ap.ArgumentParser()
parser.add_argument("--src-dir", type=str, help="path to original mask dir")
parser.add_argument("--dst-dir", type=str, help="path to processed mask dir")
args = parser.parse_args()
src_dir = Path(args.src_dir)
dest_dir = Path(args.dst_dir)

ct_ids = sorted(os.listdir(src_dir))
for ct_id in tqdm(ct_ids, desc="CT scan"):
    ct_dir = dest_dir / ct_id
    ct_dir.mkdir(parents=True, exist_ok=True)
    seg_path = glob.glob(f"{str(src_dir)}/{ct_id}/*.nii.gz")[0]
    seg_file = nib.load(seg_path)

    masks = np.array(seg_file.dataobj, dtype=np.int64)
    # match mask to CT image orientation
    masks = np.rot90(masks, k=1)
    # remove left, right lung information
    masks[masks > 0] = 1

    # Save each slice in an individual file
    for slice_idx in range(masks.shape[-1]):
        np.save(ct_dir / f"{slice_idx}.npy", masks[..., slice_idx])
