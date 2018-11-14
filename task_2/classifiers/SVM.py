from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""
    Little wrapper for sklearn SVM
"""


class SVM:

    def __init__(self):
        self.clf = LinearSVC(random_state=0, tol=1e-5)

    def fit(self, x, y):
        self.clf.fit(x, y)

    def check_accuracy(self, x, y):
        pred_labels = self.clf.predict(x)

        return accuracy_score(y, pred_labels)
