import argparse as ap
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data import NSCLCDataset, simple_collate_fn, get_common_ids
from src.preprocess import Rescale, Clip, ToTensor, Normalize
from src.models import UNet


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    # preprocessing routine
    transform_list = []
    if args.input_size != 512:
        transform_list.append(Rescale(args.input_size))
    transform_list.extend(
        [Clip(-1000, 1000), ToTensor(), Normalize(mean=[0.0], std=[1000.0]),]
    )
    transform = transforms.Compose(transform_list)

    # data prep
    # train/val split
    ct_ids = get_common_ids(
        "data/raw/NSCLC-Radiomics/", "data/processed/NSCLC_ground_truths/"
    )
    train_ratio = 0.7
    val_ratio = 0.2
    num_train_scans = int(len(ct_ids) * train_ratio)
    num_val_scans = int(len(ct_ids) * val_ratio)

    train_scans = ct_ids[:num_train_scans]
    val_scans = ct_ids[num_train_scans : num_train_scans + num_val_scans]
    test_scans = ct_ids[num_train_scans + num_val_scans :]

    train_ds = NSCLCDataset(
        metadata_path="data/processed/NSCLC-Radiomics_metadata_v2.csv",
        transform=transform,
        ct_ids=train_scans,
    )
    val_ds = NSCLCDataset(
        metadata_path="data/processed/NSCLC-Radiomics_metadata_v2.csv",
        transform=transform,
        ct_ids=val_scans,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=simple_collate_fn,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=simple_collate_fn,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # train model
    ckpt_path = f"models/unet-{args.input_size}"
    ckpt_path += "-{epoch}-{dice_coeff_val:.3f}"
    ckpt_callback = ModelCheckpoint(
        filepath=ckpt_path,
        monitor="dice_coeff_val",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="dice_coeff_val", mode="max", patience=args.patience, verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        gpus=[0],
        checkpoint_callback=ckpt_callback,
        early_stop_callback=early_stop_callback,
        resume_from_checkpoint=args.checkpoint,
    )
    net = UNet(in_c=1, num_classes=2)

    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
