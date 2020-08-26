import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from src.metrics import dice_coeff_from_logits


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # 'same' padding
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pipeline(x)


class PoolDoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(PoolDoubleConv, self).__init__()
        self.pipeline = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), DoubleConv(in_c, out_c)
        )

    def forward(self, x):
        return self.pipeline(x)


class UpsampleDoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpsampleDoubleConv, self).__init__()
        # Halving no. out channels, since we will
        # concat inputs from the skip connection
        self.upsample = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip_x):
        x = self.upsample(x)
        # cropping
        dh = skip_x.shape[2] - x.shape[2]
        dw = skip_x.shape[3] - x.shape[3]
        # note: pad=(padding_left,padding_right,padding_top,padding_bottom)
        x = F.pad(x, pad=[dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip_x, x], dim=1)
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self, in_c, num_classes):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        self.first_conv = DoubleConv(in_c, 64)
        self.down_blocks = nn.ModuleList([
            PoolDoubleConv(64, 128),
            PoolDoubleConv(128, 256),
            PoolDoubleConv(256, 512),
            PoolDoubleConv(512, 1024),
        ])
        self.up_blocks = nn.ModuleList([
            UpsampleDoubleConv(1024, 512),
            UpsampleDoubleConv(512, 256),
            UpsampleDoubleConv(256, 128),
            UpsampleDoubleConv(128, 64),
        ])
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.first_conv(x)
        skip_xs = [x]
        # contracting path
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            if i != len(self.down_blocks) - 1:
                skip_xs.insert(0, x)
        # expansive path
        for skip_x, up in zip(skip_xs, self.up_blocks):
            x = up(x, skip_x)
        x = self.final_conv(x)
        return x

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return opt

    def training_step(self, batch, batch_idx):
        X, y = batch["img"], batch["mask"]
        logits = self(X)
        loss = self.loss_fn(logits, y)

        # evaluation metric
        dice = dice_coeff_from_logits(logits, y)
        pbar = {"dice_coeff": dice}

        return {
            "loss": loss,
            "progress_bar": pbar,
            "log": {"loss/train": loss, "dice_coeff/train": dice},
        }

    def validation_step(self, batch, batch_idx):
        # reuse forward pass from training step
        res = self.training_step(batch, batch_idx)
        return res

    def validation_epoch_end(self, val_step_outputs, return_logs=True):
        avg_val_loss = torch.tensor([res["loss"] for res in val_step_outputs]).mean()
        avg_dice_coeff = torch.tensor(
            [res["progress_bar"]["dice_coeff"] for res in val_step_outputs]
        ).mean()

        res = {
            "val_loss": avg_val_loss,
            "progress_bar": {"avg_val_dice_coeff": avg_dice_coeff},
        }
        if return_logs:
            res["log"] = {
                "loss/val": avg_val_loss,
                "dice_coeff/val": avg_dice_coeff,
                "dice_coeff_val": avg_dice_coeff,  # avoid slash in ckpt file name
            }
        return res

    def test_step(self, batch, batch_idx):
        # reuse forward pass from training step
        res = self.training_step(batch, batch_idx)
        return res

    def test_epoch_end(self, test_step_outputs):
        # reuse valid stat accumulation
        res = self.validation_epoch_end(test_step_outputs, return_logs=False)
        test_dsc = res["progress_bar"]["avg_val_dice_coeff"]
        print(test_dsc)
        return res
