import argparse as ap
from pathlib import Path
import yaml

import pytorch_lightning as pl
from torch.utils.data import Subset

from src.models import UNet
from src.data import PlethoraDataModule, KmaderDataModule, Covid19DataModule


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--dataset", type=str, default="plethora")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, nargs="+")

    args = parser.parse_args()

    data_params_path = str(Path(args.checkpoint).parents[1] / "data_hparams.yaml")
    with open(data_params_path, "r") as f:
        data_module_args = yaml.safe_load(f)
    # override batch size
    data_module_args["batch_size"] = args.batch_size
    if args.dataset == "plethora":
        data_module = PlethoraDataModule(**data_module_args)
    elif args.dataset == "kmader":
        data_module = KmaderDataModule(**data_module_args)
    elif args.dataset == "covid19":
        data_module = Covid19DataModule(**data_module_args)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}, can only be one of: [plethora, kmader]")
    data_module.setup("test")

    if args.model == "unet":
        net = UNet.load_from_checkpoint(args.checkpoint)

    print(f"evaluating {args.model} on {args.dataset} dataset, batch_size={args.batch_size}")
    trainer = pl.Trainer(logger=False, gpus=args.gpus)
    trainer.test(net, datamodule=data_module)
