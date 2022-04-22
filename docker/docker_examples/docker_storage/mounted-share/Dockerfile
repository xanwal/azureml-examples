FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

USER root 

RUN  apt-get update
RUN  apt-get install -y wget apt-utils 
RUN  wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
RUN  dpkg -i packages-microsoft-prod.deb 
RUN  apt-get remove -y wget 
RUN  apt-get update 
RUN  apt-get install -y --no-install-recommends fuse blobfuse libcurl3-gnutls libgnutls30 
RUN rm -rf /var/lib/apt/lists/*

USER dockeruser

ENV BFMOUNT=/home/dockeruser/blobfuse
RUN mkdir ${BFMOUNT} && chmod 0755 ${BFMOUNT} 
ENV BFCONFMOUNT=/etc/bf/bfconf

ENTRYPOINT blobfuse ${BFMOUNT} --use-https=true --config-file=/home/fuse_connection.cfg --tmp-path=/tmp/blobfuse/tmp -o allow_other


