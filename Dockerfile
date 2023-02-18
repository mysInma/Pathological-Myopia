FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade --quiet jupyter_client ipywidgets
WORKDIR /workspace/notebooks