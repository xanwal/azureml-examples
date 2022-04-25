FROM "mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest"

USER root
RUN apt-get update
RUN apt-get install -y p7zip-full libpng-dev

USER dockeruser

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

