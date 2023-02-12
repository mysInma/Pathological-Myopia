curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=ubuntu20.04
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
apt-get install nvidia-container-runtime
apt -y install nvidia-container-toolkit 
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker 
