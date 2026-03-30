import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """Initialize early-stopping hyperparameters and tracking state.

        Args:
            patience: Maximum epochs without validation improvement.
            delta: Minimum validation improvement to reset patience.
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
        """Update early-stopping state from the current validation loss.

        Args:
            val_loss: Current validation loss.
            model: SBBTS drift model.
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
        """Load the best checkpointed model weights into the provided model.

        Args:
            model: SBBTS drift model.
        """
        model.load_state_dict(self.best_model_state)
