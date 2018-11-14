from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RF:

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=50)

    def fit(self, x, y):
        self.clf.fit(x, y)

    def check_accuracy(self, x, y):
        pred_labels = self.clf.predict(x)

        accuracy = accuracy_score(y, pred_labels)

        return accuracy