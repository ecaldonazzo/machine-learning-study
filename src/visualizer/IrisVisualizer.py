import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class IrisVisualizer:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_confusion_matrix(self, save_path=None):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusão - Iris")
        plt.xlabel("Previsto")
        plt.ylabel("Real")

        if save_path:
            # garante que o diretório existe antes de salvar
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        plt.show()
