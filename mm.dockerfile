FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ENV TZ Asia/Shanghai
LABEL maintainer="goog"

# WORKDIR /ssd/xianyiliu
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update

RUN apt-get install -y g++ 

RUN pip install dgl -i https://pypi.douban.com/simple
RUN pip install visdom -i https://pypi.douban.com/simple
RUN pip install scikit-learn -i https://pypi.douban.com/simple
RUN pip install tensorboard
RUN pip install pandas
RUN pip install lmdb
RUN pip install matplotlib 

RUN pip install torch_geometric
RUN pip install torch_scatter