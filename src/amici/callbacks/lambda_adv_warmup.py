import pytorch_lightning as pl

class LambdaAdvWarmupCallback(pl.Callback):
    def __init__(self, warmup_epochs=20, max_val=0.2):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_val = max_val

    def on_train_epoch_start(self, trainer, pl_module):
        """Update lambda_adv at the start of each epoch."""
        epoch = trainer.current_epoch
        lam = self.max_val * min(1.0, epoch / self.warmup_epochs)

        if hasattr(pl_module, "module"):
            pl_module.module.lambda_adv = lam
        else:
            pl_module.lambda_adv = lam

        # Optional logging
        trainer.logger.log_metrics({"lambda_adv": lam}, step=epoch)
