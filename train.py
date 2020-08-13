import torch
from torch.utils.data import DataLoader, Subset, random_split
import pytorch_lightning as pl

from src.data import NSCLCDataset, simple_collate_fn
from src.preprocess import DEFAULT_TRANSFORM
from src.models import UNet


# hyperparams
batch_size = 16
train_ratio = 0.8

# preprocessing routine
transform = DEFAULT_TRANSFORM

# data prep
full_ds = NSCLCDataset(metadata_path="data/processed/NSCLC-Radiomics_metadata.csv",
                       transform=transform)
# train/val split
num_train = int(train_ratio * len(full_ds))
num_val = len(full_ds) - num_train
train_ds, val_ds = random_split(full_ds, lengths=[num_train, num_val],
                                generator=torch.Generator().manual_seed(25))
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    collate_fn=simple_collate_fn,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    collate_fn=simple_collate_fn,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
)

# train model
net = UNet(in_c=1, num_classes=2)
trainer = pl.Trainer(max_epochs=50, gpus=1)
trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
