FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Install packages
RUN rm -f /etc/apt/sources.list.d/cuda.list \
	&& apt-get update && apt-get install -y --no-install-recommends \
	wget \
	&& distro=$(. /usr/lib/os-release; echo $ID$VERSION_ID | tr -d ".") \
	&& arch=$(/usr/bin/arch) \
	&& wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb \
	&& dpkg -i cuda-keyring_1.0-1_all.deb \
	&& rm -f cuda-keyring_1.0-1_all.deb
RUN apt-get update \
    && apt-get install -y sudo wget zip vim git
RUN apt-get install -y libxrender1

# Setup conda
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.10.0-1-Linux-x86_64.sh \
    && bash /opt/Miniconda3-py39_23.10.0-1-Linux-x86_64.sh -b -u -p /opt/miniconda3 \
    && rm -f /opt/Miniconda3-py39_23.10.0-1-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

# Create conda env
ARG env_name=drone
RUN conda create -yn ${env_name} python=3.10 \
	&& conda init
ENV CONDA_DEFAULT_ENV ${env_name}
RUN echo "conda activate ${env_name}" >> ~/.bashrc
ENV PATH /opt/miniconda3/envs/${env_name}/bin:$PATH

# Install requirements
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
