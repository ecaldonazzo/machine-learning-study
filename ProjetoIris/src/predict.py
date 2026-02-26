import joblib
import numpy as np

# Carregar modelo treinado
knn = joblib.load("../models/knn_iris.pkl")

# Exemplo de nova flor: [sepal length, sepal width, petal length, petal width]
nova_flor = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = knn.predict(nova_flor)

# Mapear para nomes das espécies
classes = {0: "setosa", 1: "versicolor", 2: "virginica"}
print("Espécie prevista:", classes[pred[0]])