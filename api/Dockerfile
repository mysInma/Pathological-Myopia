FROM python:3.9-bullseye
COPY ./requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace/fast-api
ENTRYPOINT ["tail", "-f", "/dev/null"]