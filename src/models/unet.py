import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from src.metrics import dice_coeff_vectorized
from src.losses import SoftDiceLoss


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
        dh = skip_x.shape[2] - x.shape[2]
        dw = skip_x.shape[3] - x.shape[3]
        # note: pad=(padding_left,padding_right,padding_top,padding_bottom)
        x = F.pad(x, pad=[dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip_x, x], dim=1)
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self, in_c, num_classes,
                 loss='cross_entropy'):
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(1, 1, 512, 512)

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

        if loss == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == "soft_dice":
            self.loss_fn = SoftDiceLoss(num_classes)
        else:
            raise NotImplementedError(
                f"Unknown loss: {loss}, can only be one of: [cross_entropy, soft_dice]")

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

    def forward_pass(self, batch):
        if self.loss_fn is None:
            raise RuntimeError("loss function was not specified")

        X, y = batch["img"], batch["mask"]
        logits = self(X)
        loss = self.loss_fn(logits, y)

        # evaluation metric
        pred_masks = torch.argmax(logits, dim=1)
        dsc = dice_coeff_vectorized(pred_masks, y, reduce_fn=torch.mean)

        return loss, dsc

    def training_step(self, batch, batch_idx):
        loss, dsc = self.forward_pass(batch)

        result = pl.TrainResult(minimize=loss)
        result.log("loss/train", loss)
        result.log("dice_coeff/train", dsc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss, dsc = self.forward_pass(batch)

        result = pl.EvalResult(early_stop_on=dsc, checkpoint_on=dsc)
        result.log("loss/val", loss)
        result.log("dice_coeff/val", dsc, prog_bar=True, on_step=False,
                   on_epoch=True, reduce_fx=torch.mean)
        result.log("dice_coeff_val", dsc, prog_bar=False, on_step=False,
                   on_epoch=True, reduce_fx=torch.mean)
        return result

    def on_test_epoch_start(self):
        self.true_buffer = []
        self.pred_buffer = []

    def test_step(self, batch, batch_idx):
        X, y = batch["img"], batch["mask"]
        logits = self(X)
        pred_masks = torch.argmax(logits, dim=1)
        dsc = dice_coeff_vectorized(pred_masks, y, reduce_fn=torch.mean)

        result = pl.EvalResult()
        result.log("dice_2d", dsc, prog_bar=True, on_step=True, on_epoch=False)

        # 3D dice coeff
        slice_idxs = batch["slice_idx"]
        split_idx = torch.where(slice_idxs == 0)[0]
        if len(split_idx) > 1:
            raise RuntimeError(f"there are multiple zeros in slice_idxs: {slice_idxs}")
        split_idx = split_idx.item() if len(split_idx) > 0 else None
        self.true_buffer.append(y[:split_idx])
        self.pred_buffer.append(pred_masks[:split_idx])

        # if we have finish iterating over a CT scan, calculate 3D dice
        if (split_idx is not None and batch_idx > 0) or \
                batch_idx == len(self.test_dataloader()) - 1:
            true_v_mask = torch.cat(self.true_buffer).reshape(1, -1)
            pred_v_mask = torch.cat(self.pred_buffer).reshape(1, -1)

            dsc_v = dice_coeff_vectorized(pred_v_mask, true_v_mask, reduce_fn=None)
            result.log("dice_3d", dsc_v, prog_bar=True, on_step=True, on_epoch=False)

            # clear the buffers
            self.true_buffer = [y[split_idx:]]
            self.pred_buffer = [pred_masks[split_idx:]]
            del true_v_mask
            del pred_v_mask

        return result

    def test_epoch_end(self, test_step_outputs):
        mean_dice_2d = test_step_outputs["dice_2d"].mean()
        mean_dice_3d = test_step_outputs["dice_3d"].mean()

        result = pl.EvalResult()
        result.log("mean_dice_2d", mean_dice_2d)
        result.log("mean_dice_3d", mean_dice_3d)
        return result
