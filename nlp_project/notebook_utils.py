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


def split(x, y, test_size=0.2, val_size=0.0):
    if val_size + test_size >= 1:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size + val_size, stratify=y
    )
    x_val, y_val = None, None
    if val_size > 0:
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=val_size, stratify=y_test
        )
    return x_train, x_val, x_test, y_train, y_val, y_test


def evaluate(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.show()
