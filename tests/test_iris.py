import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ─── SimpleDatasetLoader ────────────────────────────────────────────────────

class TestSimpleDatasetLoader:
    def setup_method(self):
        from src.data_handler.SimpleDatasetLoader import SimpleDatasetLoader
        self.handler = SimpleDatasetLoader(test_size=0.2)

    def test_load_returns_four_splits(self):
        X_train, X_test, y_train, y_test = self.handler.load_sklearn("iris")
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = self.handler.load_sklearn("iris")
        total = len(X_train) + len(X_test)
        assert total == 150  # Iris tem 150 amostras

    def test_test_size_proportion(self):
        X_train, X_test, y_train, y_test = self.handler.load_sklearn("iris")
        assert len(X_test) == 30   # 20% de 150
        assert len(X_train) == 120  # 80% de 150

    def test_feature_count(self):
        X_train, X_test, y_train, y_test = self.handler.load_sklearn("iris")
        assert X_train.shape[1] == 4  # 4 features no Iris

    def test_classes(self):
        X_train, X_test, y_train, y_test = self.handler.load_sklearn("iris")
        all_y = np.concatenate([y_train, y_test])
        assert len(np.unique(all_y)) == 3  # 3 classes


# ─── KnnTrainer ─────────────────────────────────────────────────────────────

class TestKnnTrainer:
    def setup_method(self):
        from src.data_handler.SimpleDatasetLoader import SimpleDatasetLoader
        from src.model_trainer.KnnTrainer import KnnTrainer
        handler = SimpleDatasetLoader(test_size=0.2)
        self.X_train, self.X_test, self.y_train, self.y_test = handler.load_sklearn("iris")
        self.trainer = KnnTrainer(n_neighbors=5)

    def test_train_creates_model(self):
        self.trainer.train(self.X_train, self.y_train)
        assert self.trainer.model is not None

    def test_predict_returns_correct_length(self):
        self.trainer.train(self.X_train, self.y_train)
        y_pred = self.trainer.predict(self.X_test)
        assert len(y_pred) == len(self.y_test)

    def test_predict_valid_classes(self):
        self.trainer.train(self.X_train, self.y_train)
        y_pred = self.trainer.predict(self.X_test)
        assert set(y_pred).issubset({0, 1, 2})


# ─── Evaluator ──────────────────────────────────────────────────────────────

class TestEvaluator:
    def setup_method(self):
        from src.data_handler.SimpleDatasetLoader import SimpleDatasetLoader
        from src.model_trainer.KnnTrainer import KnnTrainer
        from src.evaluator.Evaluator import Evaluator
        handler = SimpleDatasetLoader(test_size=0.2)
        X_train, X_test, y_train, y_test = handler.load_sklearn("iris")
        trainer = KnnTrainer(n_neighbors=5)
        trainer.train(X_train, y_train)
        y_pred = trainer.predict(X_test)
        self.evaluator = Evaluator(y_test, y_pred)

    def test_accuracy_above_90(self):
        assert self.evaluator.accuracy() >= 0.90

    def test_accuracy_between_0_and_1(self):
        acc = self.evaluator.accuracy()
        assert 0.0 <= acc <= 1.0

    def test_report_not_empty(self):
        report = self.evaluator.report()
        assert report is not None
        assert len(str(report)) > 0

    def test_confusion_matrix_shape(self):
        matrix = self.evaluator.confusion()
        assert matrix.shape == (3, 3)  # 3 classes → matriz 3x3


# ─── IrisVisualizer ─────────────────────────────────────────────────────────

class TestIrisVisualizer:
    def setup_method(self):
        from src.data_handler.SimpleDatasetLoader import SimpleDatasetLoader
        from src.model_trainer.KnnTrainer import KnnTrainer
        handler = SimpleDatasetLoader(test_size=0.2)
        X_train, X_test, y_train, y_test = handler.load_sklearn("iris")
        trainer = KnnTrainer(n_neighbors=5)
        trainer.train(X_train, y_train)
        self.y_test = y_test
        self.y_pred = trainer.predict(X_test)

    def test_confusion_matrix_saved(self, tmp_path):
        from src.visualizer.IrisVisualizer import IrisVisualizer
        save_path = str(tmp_path / "confusion_matrix.png")
        viz = IrisVisualizer(self.y_test, self.y_pred)
        viz.plot_confusion_matrix(save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_output_dir_created_automatically(self, tmp_path):
        from src.visualizer.IrisVisualizer import IrisVisualizer
        save_path = str(tmp_path / "subdir" / "nested" / "matrix.png")
        viz = IrisVisualizer(self.y_test, self.y_pred)
        viz.plot_confusion_matrix(save_path)
        assert os.path.exists(save_path)
