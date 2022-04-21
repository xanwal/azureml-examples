FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

RUN apt install python3, python3-pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt