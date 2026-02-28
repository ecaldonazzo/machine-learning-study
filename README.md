# Projetos de Machine Learning

Este repositório reúne projetos de estudo em **Machine Learning**, desenvolvidos para praticar conceitos de modelagem, avaliação e visualização de dados.

## Projetos

### 1. Treinamento de Modelo Iris
- **Objetivo**: Treinar e avaliar um modelo KNN usando o dataset Iris.
- **Arquitetura**:
  - `src/data_handler` → carregamento de dados
  - `src/model_trainer` → treinamento do modelo
  - `src/evaluator` → métricas de desempenho
  - `src/visualizer` → visualização da matriz de confusão
  - `src/projects` → orquestrador (`IrisProject`)
  - `IrisTrainerMain.py` → entrypoint do projeto
- **Saídas**:
  - Relatório de métricas no console
  - Matriz de confusão salva em `report/iris_confusion.png`
  - Modelo persistido em `models/knn_iris.pkl`

## Tecnologias
- Python 3.14
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- joblib
- Docker + docker-compose

## Execução
```bash
docker-compose up --build
