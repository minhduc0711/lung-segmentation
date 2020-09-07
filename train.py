import argparse as ap

import torch
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import PlethoraDataModule, KmaderDataModule
from src.models import UNet


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument("--model", type=str, default="unet",
                        )
    parser.add_argument("--dataset", type=str, default="plethora",
                        help="dataset to use, one of ['plethora', 'kmader']")

    parser.add_argument("--epochs", type=int, default=50,
                        help="maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--clip-low", type=float, default=-1000)
    parser.add_argument("--clip-high", type=float, default=1000)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--exp-version", type=int, default=None)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    # data prep
    data_module_args = {
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "pin_memory": True,
        "num_workers": 4
    }
    if args.dataset == "plethora":
        data_module = PlethoraDataModule(**data_module_args)
    elif args.dataset == "kmader":
        data_module = KmaderDataModule(**data_module_args)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    data_module.setup("fit")

    # make sure that all models on multi gpus have the same weights
    pl.seed_everything(25)

    # train model
    if args.model == "unet":
        net = UNet(in_c=data_module.dims[0],
                   num_classes=data_module.num_classes)
    else:
        raise RuntimeError("Unknown model: {args.model}")

    exp_name = f"{args.model}-{args.dataset}-{args.img_size}"

    logger = TensorBoardLogger(save_dir="logs", name=exp_name,
                               version=args.exp_version)
    # the 2nd string's formatting is handled by lightning
    # TODO: fix this when the filepath issue is resolved: https://github.com/PyTorchLightning/pytorch-lightning/issues/3254
    ckpt_path = f"{logger.log_dir}/ckpts/" + \
        "{epoch}-{dice_coeff_val:.3f}"
    ckpt_callback = ModelCheckpoint(
        monitor="val_checkpoint_on",
        filepath=ckpt_path,
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    # metric to monitor is defined in the LightningModule subclass
    early_stop_callback = EarlyStopping(
        mode="max", patience=args.patience, verbose=True,
    )

    distributed_backend = None
    if args.gpus is None:
        print("training on cpu")
    elif args.gpus[0] == -1:
        # select all gpus
        args.gpus = -1
        distributed_backend = "ddp"
    elif len(args.gpus) > 1:
        # multi gpus
        distributed_backend = "ddp"
    print(f"distributed: {distributed_backend}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        checkpoint_callback=ckpt_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
        resume_from_checkpoint=args.checkpoint,
        gpus=args.gpus,
        distributed_backend=distributed_backend,
        profiler=args.profile
    )

    print(f"Training {args.model} on {args.dataset} dataset.")
    print(f"Train on {len(data_module.train_ds)} samples, ", end="")
    print(f"validate on {len(data_module.val_ds)} samples.")
    trainer.fit(net, datamodule=data_module)
