# Pathological-Myopia

### Getting started on Linux (Ubuntu)

Execute the script `nvidia-container-runtime-script.sh`

To verify the instalation execute the command:

```
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.3.0-devel-ubuntu20.04 nvidia-smi
```

### Getting started on Windows

First you need to verify if nvidia drivers are instaled. Once they are instaled with **wsl**  `Ubuntu-20.04` execute script `nvidia-container-runtime-win.sh`.

Finally copy the above json and paste it on `Docker Engine` option inside the settings. This json is the docker `daemon.json` file to configure Docker Desktop. 

```
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "debug": false,
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "insecure-registries": [],
  "registry-mirrors": [],
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

To verify the instalation execute the command:

```
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.3.0-devel-ubuntu20.04 nvidia-smi
```

### Set up to develop

The current repository uses docker compose to forget about dependencies. In order to develop just type `docker-compose up -d`. If there is any change on `Dockerfile` you need to rebuild the image, but you can do it on docker compose with the following command:

```
docker-compose up -d --no-deps --build <service_name>
```
