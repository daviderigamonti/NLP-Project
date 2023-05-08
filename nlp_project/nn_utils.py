import torch
import numpy as np

history_metrics = {
    "epoch": {},
    "loss": {"order": -1},
    "val_loss": {"order": -1},
    "acc": {"order": +1},
    "val_acc": {"order": +1},
}


class StopNNLoop(BaseException):
    pass


def build_history_string(history_point):
    epoch = history_point["epoch"]
    metrics_string = " ".join(
        [f"{k}: {history_point[k]:.7f}" for k in history_point if not k == "epoch"]
    )
    return f"Epoch {epoch} -- " + metrics_string


def compare_equal_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
    return False


# Returns true if a is "better" than b following the metric
def compare_metric(metric, a, b, delta=0):
    if a == b:
        return False
    if history_metrics[metric]["order"] == +1:
        return a > b + delta
    return a < b - delta


# Initializes lowest possible value given a metric
def init_lowest(metric):
    return -np.inf if history_metrics[metric]["order"] == +1 else np.inf


def init_gpu(gpu="cuda:0"):
    return torch.device(gpu if torch.cuda.is_available() else "cpu")
