import torch
import torch.nn as nn
import torch.utils.data as dt

import numpy as np

from nlp_project.nn_utils import StopNNLoop, build_history_string


class Data(dt.Dataset):
    def __init__(self, x, y, x_type=np.int32, y_type=torch.float):
        x_coo = x.tocoo()
        self.x = torch.sparse.FloatTensor(
            torch.LongTensor([x_coo.row, x_coo.col]),
            torch.FloatTensor(x_coo.data.astype(x_type)),
            x_coo.shape,
        )
        self.y = torch.tensor(y, dtype=y_type)
        self.shape = self.x.shape

    def __getitem__(self, index):
        return self.x[index].to_dense(), self.y[index]

    def __len__(self):
        return self.shape[0]


class Classifier(nn.Module):
    def __init__(self, binary_classifier=False, device=torch.device("cpu"), verbose=True):
        super().__init__()
        self.device = device
        self.is_binary = binary_classifier
        self.verbose = verbose
        self.is_compiled = False
        self.history = []

    def forward(self, x):
        return x

    def compile(self, loss, optimizer, binary_threshold=0.5):
        self.loss = loss
        self.optimizer = optimizer
        self.binary_threshold = binary_threshold
        self.to(self.device)
        self.is_compiled = True

    def parse_logits(self, outputs):
        if self.is_binary:
            predicted = (outputs > self.binary_threshold).float()
        else:
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def train_loop(self, data, epochs, data_val=None, callbacks=[]):
        try:
            tot = len(data.dataset)
            # Iterate over all epochs
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                history_point = {}
                # Iterate over each dataset batch
                for i, datum in enumerate(data):
                    # Decompose batch in x and y
                    inputs, labels = datum
                    # Set gradients to zero
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs = self(inputs)
                    predictions = self.parse_logits(outputs)
                    current_loss = self.loss(outputs, labels)
                    # Backpropagation
                    current_loss.backward()
                    # Optimization
                    self.optimizer.step()
                    # Update metrics
                    running_loss += current_loss.item()
                    correct += (predictions == labels).float().sum()

                # Compute training metrics
                history_point["epoch"] = epoch + 1
                history_point["loss"] = running_loss / tot
                history_point["acc"] = correct / tot

                # Compute and save eventual validation metrics
                if data_val:
                    _, val_metrics = self.test_loop(data_val)
                    history_point["val_loss"] = val_metrics["loss"]
                    history_point["val_acc"] = val_metrics["acc"]

                # Save epoch in history
                self.history.append(history_point)

                # Perform callbacks
                for callback in callbacks:
                    callback.call(self, history_point)

                # Print epoch summary
                if self.verbose:
                    print(build_history_string(history_point))

        except StopNNLoop as s:  # noqa
            pass

    def test_loop(self, data):
        all_predictions = np.array([])
        tot = len(data.dataset)
        loss = 0.0
        correct = 0
        metrics = {}
        # Prevent model update
        with torch.no_grad():
            # Iterate over each dataset batch
            for datum in data:
                # Decompose batch in x and y
                inputs, labels = datum
                # Forward pass
                outputs = self(inputs)
                predictions = self.parse_logits(outputs)
                current_loss = self.loss(outputs, labels)
                # Update metrics
                loss += current_loss.item()
                correct += (predictions == labels).float().sum()
                # Append predictions
                all_predictions = np.append(all_predictions, predictions)

        # Compute metrics
        metrics["acc"] = correct / tot
        metrics["loss"] = loss / tot

        return all_predictions.flatten(), metrics
