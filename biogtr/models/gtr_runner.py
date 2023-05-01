import torch
from pytorch_lightning import LightningModule


class GTRRunner(LightningModule):
    def __init__(self, model, loss, optim=None, scheduler=None):
        super().__init__()

        self.model = model
        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler

    def training_step(self, train_batch, batch_idx):
        # todo: add logic for wandb logging

        x = train_batch[0]

        logits = self.model(x)

        loss = self.loss(logits, x)

        self.log("train_loss", loss)

        return loss

    # def validation_step(self, val_batch, batch_idx):
    # to implement. also need switch count logic
    # return loss

    def configure_optimizers(self):
        # todo: init from config
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-4, betas=(0.9, 0.999)
            )
        else:
            optimizer = self.optimizer

        if self.scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.5, 10
            )
        else:
            scheduler = self.scheduler

        return [optimizer], [scheduler]
