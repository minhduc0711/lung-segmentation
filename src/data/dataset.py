import glob
import os
import pickle

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from tqdm import tqdm

from torch.utils.data import Dataset


class NSCLCDataset(Dataset):
    def __init__(self, ct_dir=None, mask_dir=None,
                 metadata_path=None,
                 ct_ids=None,
                 hu_range=(-1000, 1000),
                 transform=None,
                 subset=None, train_ratio=0.8):
        if metadata_path:
            with open(metadata_path, "rb") as f:
                self.idx_2_ct = pickle.load(f)
        else:
            assert ct_dir and mask_dir, \
                "You have to provide either metadata_path or ct_dir & mask_dir"
            if not ct_ids:
                # only include CT scans with corresponding masks available
                img_ids = set(os.listdir(ct_dir))
                seg_ids = set(os.listdir(mask_dir))
                ct_ids = list(img_ids.intersection(seg_ids))
            num_scans_train = int(len(ct_ids) * train_ratio)

            if subset == "train":
                ct_ids = ct_ids[:num_scans_train]
            elif subset == "val":
                ct_ids = ct_ids[num_scans_train:]
            elif subset is None:
                pass
            else:
                raise RuntimeError(f"Subset {subset} not found")

            self.cache_metadata(ct_dir, mask_dir, ct_ids, subset)
        self.hu_range = hu_range
        self.transform = transform

    def cache_metadata(self, ct_dir, mask_dir, ct_ids, subset):
        idx_2_ct = {}
        cumu_num_slices = 0
        for ct_id in tqdm(ct_ids, desc="Caching CT scans metadata"):
            dicom_paths = glob.glob(f"{ct_dir}/{ct_id}/*/*/*/*.dcm")
            dicom_paths = sorted(dicom_paths,
                                 key=lambda p: pydicom.read_file(p).SliceLocation)
            num_slices = len(dicom_paths)
            seg_path_str = "{}/{}/{}.npy"
            # store a dict of:
            # sample_idx -> (CT slice idx, CT slice dicom path, seg mask path)
            for slice_idx in range(num_slices):
                sample_idx = cumu_num_slices + slice_idx

                dicom_path = os.path.abspath(dicom_paths[slice_idx])
                seg_path = seg_path_str.format(mask_dir, ct_id, slice_idx)
                seg_path = os.path.abspath(seg_path)

                idx_2_ct[sample_idx] = (slice_idx, dicom_path, seg_path)
            cumu_num_slices += num_slices

        # save metadata dict to disk
        dataset_name = os.path.basename(os.path.normpath(ct_dir))
        if subset:
            dict_path = f"data/processed/{dataset_name}_{subset}_metadata.pkl"
        else:
            dict_path = f"data/processed/{dataset_name}_metadata.pkl"
        with open(dict_path, "wb") as f:
            pickle.dump(idx_2_ct, f, pickle.HIGHEST_PROTOCOL)
        self.idx_2_ct = idx_2_ct

    def __len__(self):
        return len(self.idx_2_ct)

    def __getitem__(self, idx):
        slice_idx, dicom_path, seg_path = self.idx_2_ct[idx]

        ds = pydicom.read_file(dicom_path)
        # convert to HU scale
        ct_slice = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        # add explicit channel dim to images
        ct_slice = np.expand_dims(ct_slice, axis=-1)
        seg_mask = np.load(seg_path)

        sample = {"ct_slice": ct_slice, "seg_mask": seg_mask}
        if self.transform:
            sample = self.transform(sample)

        return sample
