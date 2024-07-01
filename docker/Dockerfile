FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN sed -i -e 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.ustc\.edu\.cn\/ubuntu/' /etc/apt/sources.list

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y git curl wget tree parallel htop unzip sudo vim tzdata python3 python3-venv python3-wheel python3-distutils python3-dev strace rename openssh-server rsync libglfw3-dev

COPY --from=continuumio/miniconda3:22.11.1 /opt/conda /opt/conda

RUN mkdir /dep
WORKDIR /dep
RUN chmod 777 /dep

RUN cd /dep

# prepare Mujoco
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
# # ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
# # ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

RUN /opt/conda/bin/python3.10 -m pip install wheel 'cython<3' --index https://mirrors.ustc.edu.cn/pypi/web/simple
RUN /opt/conda/bin/python3.10 -m pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu118

RUN /opt/conda/bin/python3.10 -m pip install "jax[cuda11_pip]==0.4.25" "nvidia-cudnn-cu11<9" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --index https://mirrors.ustc.edu.cn/pypi/web/simple

# Custom Installation

COPY ./requirements.txt /dep/requirements.txt
RUN /opt/conda/bin/python3.10 -m pip install -r ./requirements.txt --index https://mirrors.ustc.edu.cn/pypi/web/simple

RUN /opt/conda/bin/python3.10 -m pip install "jax[cuda11_pip]==0.4.25" "nvidia-cudnn-cu11<9" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --index https://mirrors.ustc.edu.cn/pypi/web/simple

RUN /opt/conda/bin/python3.10 -m pip install 'numpy<2' --index https://mirrors.ustc.edu.cn/pypi/web/simple


CMD ["bash"]
