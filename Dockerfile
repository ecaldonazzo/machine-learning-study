FROM python:3.14

WORKDIR /opt/project/PythonProjects

RUN useradd -m appuser

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p report models

RUN printf '#!/bin/sh\nchown -R appuser:appuser /opt/project/PythonProjects/report\nchown -R appuser:appuser /opt/project/PythonProjects/models\nexec gosu appuser "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "IrisTrainerMain.py"]