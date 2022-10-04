import sys

from pytorch_lightning.callbacks import Callback, EarlyStopping, ProgressBar
from tqdm import tqdm


class AugLagrangianCallback(Callback):
    """
    Callback used for adjusting the internal state of the lightning module, and successfully training with augmented lagrangian loss
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % 2 == 0:
            # will return -1 to stop training if initial patience is done
            pl_module.update_lagrangians()


class CustomProgressBar(ProgressBar):
    def init_train_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar


class ConditionalEarlyStopping(EarlyStopping):
    def _should_skip_check(self, trainer) -> bool:
        skip_ = super()._should_skip_check(trainer)
        skip_es_constraints = True
        if hasattr(trainer, "satisfied_constraints"):
            if trainer.satisfied_constraints:
                skip_es_constraints = False
        return skip_ or skip_es_constraints
