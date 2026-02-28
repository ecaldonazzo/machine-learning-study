from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KnnTrainer:
    def __init__(self, n_neighbors: int = 3) -> None:
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return self.model.score(X_test, y_test)