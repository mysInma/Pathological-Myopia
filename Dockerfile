FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY ./requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace/notebooks