import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris


class SimpleDatasetLoader:
    def __init__(self, test_size: float = 0.3, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    # Método para datasets do sklearn
    def load_sklearn(self, dataset: str = "iris") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        datasets = {
            "iris": load_iris,
        }
        if dataset not in datasets:
            raise ValueError(f"Dataset '{dataset}' não suportado. Escolha entre: {list(datasets.keys())}")

        data = datasets[dataset]()
        X, y = data.data, data.target

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # Método para CSV externo
    def load_csv(self, csv_path: str, target_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
