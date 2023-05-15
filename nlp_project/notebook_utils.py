import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils import resample

from os.path import exists, isfile
from joblib import dump, load
from pathlib import Path


def split(x, y, test_size=0.2, val_size=0.0, seed=0):
    if val_size + test_size >= 1:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size + val_size, stratify=y, random_state=seed
    )
    x_val, y_val = None, None
    if val_size > 0:
        x_test, x_val, y_test, y_val = train_test_split(
            x_test,
            y_test,
            test_size=val_size / (test_size + val_size),
            stratify=y_test,
            random_state=seed,
        )
    return x_train, x_val, x_test, y_train, y_val, y_test

def compact_split(dataset, test_size=0.2, val_size=0.0, seed=0):
    if val_size + test_size >= 1:
        return None
    train, test = train_test_split(
        dataset, test_size=test_size + val_size, random_state=seed
    )
    val = None
    if val_size > 0:
        val, test = train_test_split(
            test,
            test_size=test_size / (test_size + val_size),
            random_state=seed,
        )
    return train, val, test

def evaluate(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.show()


def save_scikit_model(path, model, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    dump(model, path + "/" + name)


def load_scikit_model(path, name):
    model_path = path + "/" + name
    if exists(model_path) and isfile(model_path):
        try:
            return load(model_path)
        except:
            pass
    return None
