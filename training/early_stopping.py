import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            patience: Early-stopping patience in epochs.
            delta: Minimum improvement threshold used by early stopping.

        Returns:
            None.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        """Evaluate one early-stopping step.

        Update the best checkpoint and stopping flag using the current validation loss.

        Args:
            val_loss: Validation loss at the current epoch.
            model: Neural network model used to estimate the SBBTS drift.

        Returns:
            None.
        """
        self.current_epoch += 1
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = self.current_epoch

        elif score + self.delta >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = self.current_epoch
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        """Load best model.

    Args:
            model: Neural network model used to estimate the SBBTS drift.

        Returns:
            None.
        """
        model.load_state_dict(self.best_model_state)
