import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.data import NSCLCDataset, simple_collate_fn
from src.preprocess import DEFAULT_TRANSFORM
from src.models import UNet


# hyperparams
batch_size = 16

# preprocessing routine
transform = DEFAULT_TRANSFORM

# data prep
train_ds = NSCLCDataset(
    metadata_path="data/processed/NSCLC-Radiomics_train_metadata.pkl",
    transform=transform,
)
val_ds = NSCLCDataset(
    metadata_path="data/processed/NSCLC-Radiomics_val_metadata.pkl",
    transform=transform,
)
# using smaller subsets for now
train_ds_sm = Subset(train_ds, range(5000))
val_ds_sm = Subset(train_ds, range(5000, 6000))
train_loader = DataLoader(
    train_ds_sm,
    batch_size=batch_size,
    collate_fn=simple_collate_fn,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds_sm,
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
