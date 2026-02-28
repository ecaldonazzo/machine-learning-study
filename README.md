# Projetos de Machine Learning e Inteligência Artificial

Este repositório reúne projetos de estudo em **Machine Learning** e **IA**, desenvolvidos para praticar conceitos de modelagem, avaliação e visualização de dados.

## Arquitetura

Os projetos seguem uma arquitetura modular organizada em camadas dentro do diretório `src`:

- **data_handler** → carregamento de dados (ex.: `SimpleDatasetLoader`)
- **model_trainer** → treinamento de modelos (ex.: `KnnTrainer`)
- **evaluator** → avaliação de desempenho (acurácia, relatório, matriz de confusão)
- **visualizer** → visualização de resultados (heatmap da matriz de confusão)
- **projects** → orquestrador que integra todas as etapas (ex.: `IrisProject`)
- **Main** → arquivo principal que dispara o fluxo (ex.: `IrisTrainerMain.py`)

## Tecnologias utilizadas
- Python 3.14
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- joblib
- Docker + docker-compose

## Execução
```bash
docker-compose up --build
