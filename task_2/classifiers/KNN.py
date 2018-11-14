from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNN:

    def __init__(self):
        self.clf = KNeighborsClassifier()

    def fit(self, x, y):
        self.clf.fit(x, y)

    def check_accuracy(self, x, y):
        pred_labels = self.clf.predict(x)

        return accuracy_score(y, pred_labels)
