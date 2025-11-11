FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

RUN git clone https://github.com/your-repo/my_project.git
RUN pip install numpy pandas && \
    rm -rf /root/.cache/pip

RUN apt-get update && \
    apt-get install -y git tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*