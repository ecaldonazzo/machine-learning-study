from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluator:
    def __init__(self, y_test, y_pred) -> None:
        self.y_test = y_test
        self.y_pred = y_pred

    def accuracy(self) -> float:
        return accuracy_score(self.y_test, self.y_pred)

    def report(self) -> str:
        return classification_report(self.y_test, self.y_pred)

    def confusion(self):
        return confusion_matrix(self.y_test, self.y_pred)