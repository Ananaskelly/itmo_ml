import xgboost
from sklearn.metrics import accuracy_score

"""
    Little wrapper for xgboost
"""


class XGBOOST:

    def __init__(self):
        self.clf = xgboost.XGBClassifier(learning_rate=0.01, n_estimators=300, max_depth=9)

    def fit(self, x, y):
        self.clf.fit(x, y)

    def check_accuracy(self, x, y):
        pred_labels = self.clf.predict(x)

        return accuracy_score(y, pred_labels)
