FROM quay.io/fenicsproject/stable:latest
RUN apt update && apt install libglapi-mesa
RUN pip3 install --upgrade pip && pip3 install osqp && pip3 install --upgrade numpy==1.19.5
WORKDIR /data
RUN wget --quiet https://www.paraview.org/files/v5.11/ParaView-5.11.0-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz
WORKDIR /usr/local
RUN tar -xzf /data/ParaView-5.11.0-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz --strip-components=1 
WORKDIR /home/fenics/shared
