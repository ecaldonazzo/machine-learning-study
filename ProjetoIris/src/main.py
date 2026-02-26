

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Carregar dataset
iris = load_iris()
 = iris['data']
y = iris['target']

# 2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3. Treinar modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. Avaliar modelo
y_pred = knn.predict(X_test)
print("Acurácia no teste:", knn.score(X_test, y_test))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("Relatório de classificação:\n", classification_report(y_test, y_pred))

# 5. Salvar modelo treinado
joblib.dump(knn, "../models/knn_iris.pkl")

# 6. Visualizações
iris_dataframe = pd.DataFrame(np.c_[X, y],
                              columns=np.append(iris['feature_names'], 'target'))

# Converter target para nomes das espécies
iris_dataframe['target'] = iris_dataframe['target'].map(
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)

# Scatter matrix
pd.plotting.scatter_matrix(
    iris_dataframe.iloc[:, :4],
    figsize=(11, 11),
    c=y,
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8
)
plt.show()

# Parallel coordinates
plt.figure()
pd.plotting.parallel_coordinates(iris_dataframe, "target")
plt.show()