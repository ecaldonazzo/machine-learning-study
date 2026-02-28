FROM python:3.14

WORKDIR /opt/project/PythonProjects

RUN mkdir -p /opt/project/PythonProjects/models \
    && mkdir -p /opt/project/PythonProjects/report

RUN pip install --no-cache-dir scikit-learn pandas numpy matplotlib seaborn joblib pytest

CMD ["python", "IrisTrainerMain.py"]