FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

# tmp 
COPY environment /var/environment

RUN pip install -r /var/environment/requirements.txt

COPY code /var/azureml-app
ENV AZUREML_ENTRY_SCRIPT=score.py

COPY model /var/azureml-app/azureml-models
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models