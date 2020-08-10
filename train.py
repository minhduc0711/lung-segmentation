import pytorch_lightning as pl
from src.models.unet import UNet

net = UNet(in_c=1, num_classes=2)
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(net)