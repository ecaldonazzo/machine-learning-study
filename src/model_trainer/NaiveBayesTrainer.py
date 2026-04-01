from sklearn.naive_bayes import GaussianNB
import numpy as np

class NaiveBayesTrainer:
    def __init__(self) -> None:
        self.model = GaussianNB()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return self.model.score(X_test, y_test)