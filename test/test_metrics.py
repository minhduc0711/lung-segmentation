import glob
import nibabel as nib
import torch
import numpy as np
from sklearn.metrics import f1_score

from src.metrics import dice_coeff, dice_coeff_vectorized

ERROR_THRESH = 1e-6


def test_dice_score():
    mask_path = glob.glob("data/raw/Thoracic_Cavities/LUNG1-001/*.nii.gz")[0]
    mask_file = nib.load(mask_path)
    mask_arr = np.array(mask_file.dataobj, dtype=np.int64)
    mask_arr[mask_arr > 0] = 1
    mask_arr = torch.from_numpy(mask_arr)

    target = mask_arr.permute(2, 0, 1)
    # shuffle the original masks
    input = target[torch.randperm(target.shape[0])]

    f1_list = []
    for y_pred, y_true in zip(input.numpy(), target.numpy()):
        y_pred, y_true = y_pred.flatten(), y_true.flatten()
        f1_list.append(f1_score(y_true, y_pred, zero_division=1))
    f1 = np.mean(f1_list)

    dsc = dice_coeff(input, target).item()
    dsc_v = dice_coeff_vectorized(input, target).item()

    assert abs(dsc - f1) < ERROR_THRESH, \
        f"true f1-score is {f1}, but dice_coeff returns {dsc}"
    assert abs(dsc_v - f1) < ERROR_THRESH, \
        f"true f1-score is {f1}, but dice_coeff_vectorized returns {dsc_v}"


test_dice_score()
