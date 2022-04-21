FROM ubuntu:bionic

USER root 

RUN  apt-get update
RUN  apt-get install -y wget apt-utils 
RUN  wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
RUN  dpkg -i packages-microsoft-prod.deb 
RUN  apt-get remove -y wget 
RUN  apt-get update 
RUN  apt-get install -y --no-install-recommends fuse blobfuse libcurl3-gnutls libgnutls30 
RUN  rm -rf /var/lib/apt/lists/*

ENV BFMOUNT=/mnt/blobfuse
ENV BFCONFMOUNT=/etc/blobfuse
ENV BFTEMPPATH=/tmp/blobfuse/tmp
RUN mkdir -p ${BFMOUNT} && \  
    mkdir -p ${BFCONFMOUNT} && \ 
    mkdir -p ${BFTEMPPATH}

COPY fuse_connection.cfg ${BFCONFMOUNT}/fuse_connection.cfg

#ENTRYPOINT blobfuse ${BFMOUNT} --use-https=true --config-file=${BFCONFMOUNT}/fuse_connection.cfg --tmp-path=/tmp/blobfuse -o allow_other


