FROM nvidia/cuda:11.3.0-base-ubuntu20.04
RUN apt update\
  && apt install -y python3 python3-pip wget git git-lfs zstd curl\
  && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y nvidia-cuda-toolkit
RUN git clone https://github.com/kingoflolz/mesh-transformer-jax.git
RUN pip3 install -r mesh-transformer-jax/requirements.txt
RUN pip3 uninstall torch -y
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install mesh-transformer-jax/ jax==0.2.12 jaxlib==0.1.68 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN git lfs install
RUN git clone https://huggingface.co/KoboldAI/GPT-J-6B-Adventure gpt-j-6B/
RUN pip3 install fastapi pydantic uvicorn && pip3 install numpy --upgrade && pip3 install git+https://github.com/huggingface/transformers
COPY web.py ./
COPY model.py ./
CMD uvicorn web:app --port 8080 --host 0.0.0.0
