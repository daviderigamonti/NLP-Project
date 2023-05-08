import abc
import copy as cp
import numpy as np

from nlp_project.nn_utils import StopNNLoop, history_metrics, compare_metric, init_lowest


class Callback(metaclass=abc.ABCMeta):
    def __init__(self, inputs):
        if not isinstance(inputs, list):
            raise TypeError("Parameter 'inputs' must be a list")
        if not all(x in history_metrics for x in inputs):
            raise ValueError(
                "Unknown input value, not present in Callback.callback_inputs"
            )
        self.inputs = inputs

    def inputs_check(self, inputs):
        if not all(x in inputs for x in self.inputs):
            raise ValueError(
                f"Requested inputs not provided: {[i for i in inputs if i not in self.inputs]}"
            )

    @abc.abstractmethod
    def call(self, model, inputs):
        self.inputs_check(inputs)
        pass


class EarlyStopping(Callback):
    def __init__(
        self,
        metric="loss",
        patience=10,
        baseline=None,
        delta=0,
        restore_best=True,
        verbose=True,
    ):
        super().__init__([metric])
        self.metric = metric
        self.patience = patience
        self.baseline = baseline
        self.delta = delta
        self.restore_best = restore_best
        self.verbose = verbose
        self.best_epoch = 0
        self.counter = 0
        self.saved_params = {}
        self.last_best = init_lowest(self.metric)

    def call(self, model, inputs):
        super().call(model, inputs)
        if self.early_stop(model, inputs):
            raise StopNNLoop()

    def early_stop(self, model, inputs):
        metric = inputs[self.metric]
        # Check if new metric is better than the current best
        if compare_metric(self.metric, metric, self.last_best):
            # Reset counter and update best value
            self.last_best = metric
            self.counter = 0
            self.best_epoch = inputs["epoch"]
            # Update model checkpoint
            if compare_metric(self.metric, metric, self.baseline):
                self.saved_params = cp.deepcopy(model.state_dict())
        # Check if new metric is worse than the current best
        elif compare_metric(self.metric, self.last_best, metric, self.delta):
            # Increment counter
            self.counter += 1
            # Check if counter exceeds patience, if so interrupt training
            if self.counter >= self.patience:
                # Restore best model checkpoint if possible and wanted
                if not self.restore_best:
                    return True
                if self.saved_params:
                    model.load_state_dict(self.saved_params)
                if self.verbose:
                    if self.saved_params:
                        print(f"Model restored successfully @ epoch {self.best_epoch}")
                    else:
                        print(f"Couldn't restore model @ epoch {self.best_epoch}")
                return True
        return False


class AdaptLR(Callback):
    def __init__(self, metric="loss", patience=5, factor=0.1, delta=0, verbose=True):
        super().__init__([metric])
        self.metric = metric
        self.patience = patience
        self.factor = factor
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.last_best = init_lowest(self.metric)

    def call(self, model, inputs):
        super().call(model, inputs)
        if self.adaptlr(inputs):
            # Adapt learning rate
            out = []
            for g in model.optimizer.param_groups:
                g["lr"] *= self.factor
                out = g["lr"]
            if self.verbose:
                print(f"Reducing lr to {out:.4f}")

    def adaptlr(self, inputs):
        metric = inputs[self.metric]
        # Check if new metric is better than the current best
        if compare_metric(self.metric, metric, self.last_best):
            # Reset counter and update best value
            self.last_best = metric
            self.counter = 0
        # Check if new metric is worse than the current best
        elif compare_metric(self.metric, self.last_best, metric, self.delta):
            # Increment counter
            self.counter += 1
            # Check if counter exceeds patience, if so interrupt training
            if self.counter >= self.patience:
                self.counter = 0
                return True
        return False
