import argparse as ap

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import NSCLCDataset, KmaderDataset, get_common_ids
from src.data.preprocess import Resize, Clip, ToTensor, Normalize, \
    ExtractMaskAroundLungs
from src.models import UNet


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--dataset", type=str, default="plethora")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--clip-low", type=float, default=-1000)
    parser.add_argument("--clip-high", type=float, default=1000)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    if args.gpus[0] == -1:
        # select all gpus
        args.gpus = -1

    # preprocessing routine
    transform_list = []
    if args.input_size != 512:
        transform_list.append(Resize(args.input_size))
    transform_list.extend([
         Clip(args.clip_low, args.clip_high),
         ExtractMaskAroundLungs(),
         ToTensor(),
         Normalize(low=0, high=1)
    ])
    transform = transforms.Compose(transform_list)

    # data prep
    if args.dataset == "plethora":
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
    elif args.dataset == "kmader":
        ds = KmaderDataset(
            "data/raw/kmader/3d_images/",
            ct_ids=["0002", "0031", "0078"],
            transform=transform,
        )
        #val_ds = KmaderDataset(
        #    "data/raw/kmader/3d_images/", ct_ids=["0059"], transform=transform
        #)
        train_ratio = 0.8
        num_train = int(train_ratio * len(ds))
        num_val = len(ds) - num_train

        train_ds, val_ds = random_split(ds, [num_train, num_val],
                                        generator=torch.Generator().manual_seed(25))
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # make sure that all models on multi gpus have the same weights
    pl.seed_everything(25)

    # train model
    if args.model == "unet":
        net = UNet(in_c=1, num_classes=2)
    else:
        raise RuntimeError("Unknown model: {args.model}")

    exp_name = f"{args.model}-{args.dataset}-{args.input_size}"

    # the 2nd string is handled by lightning formatting
    ckpt_path = f"models/{exp_name}" + "-{epoch}-{dice_coeff_val:.3f}"
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
    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        checkpoint_callback=ckpt_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
        resume_from_checkpoint=args.checkpoint,
        gpus=args.gpus,
        distributed_backend="ddp"
    )

    print(f"Training {args.model} on {args.dataset} dataset")
    print(f"Train on {len(train_ds)} samples, validate on {len(val_ds)} samples")

    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
