from src.data_handler.SimpleDatasetLoader import SimpleDatasetLoader
from src.model_trainer.NaiveBayesTrainer import NaiveBayesTrainer
from src.evaluator.Evaluator import Evaluator
from src.visualizer.IrisVisualizer import IrisVisualizer
import joblib

class NaiveBayesIrisProject:
    def run(self):
        # 1. Carregar dados
        handler = SimpleDatasetLoader(test_size=0.25)
        X_train, X_test, y_train, y_test = handler.load_sklearn("iris")

        # 2. Treinar modelo
        trainer = NaiveBayesTrainer()
        trainer.train(X_train, y_train)
        y_pred = trainer.predict(X_test)

        # 3. Avaliar
        evaluator = Evaluator(y_test, y_pred)
        print("Acurácia:", evaluator.accuracy())
        print("Relatório:\n", evaluator.report())
        print("Matriz de confusão:\n", evaluator.confusion())

        # 4. Visualizar
        viz = IrisVisualizer(y_test, y_pred)
        viz.plot_confusion_matrix("report/naive_bayes_iris_confusion.png")

        # 5. Salvar modelo
        joblib.dump(trainer.model, "models/naive_bayes_iris.pkl")