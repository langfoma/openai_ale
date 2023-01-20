# ...

## Introduction

## Setup
* Steps for creating a virtual environment.
```
python3 -m venv venv
source venv/bin/activate
```

* Basic Python module dependencies.
```
pip install --upgrade pip
pip install matplotlib opencv-python pygame gym[atari]
pip install gym==0.15.7
ale-import-roms roms
```

* Torch dependencies  
  Installation procedures for Torch will depend on whether the target workstations has a CUDA-supported GPU installed. 
  A CUDA-enabled workstation is not required. However, training neural networks with CUDA is significantly faster than 
  using CPU only. Check if you have a CUDA-supported GPU and choose the appropriate installation instructions below.
    * For non-CUDA-supported workstations (i.e., CPU-only)
        ```
        pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
        pip install torchinfo
        ```

    * For CUDA-supported workstations (i.e., GPU-supported)
        * Install CUDA 10.1
        ```
        sudo apt update
        sudo add-apt-repository ppa:graphics-drivers
        sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
        
        sudo apt update
        sudo apt install cuda-10-1
        sudo apt install libcudnn7
        ```
        * Update user profile (Update ~/.bashrc)
        ```
        # set PATH for cuda 10.1 installation
        CUDA_VER=cuda-10.1
        if [ -d "/usr/local/${CUDA_VER}/bin/" ]; then
            export PATH=/usr/local/${CUDA_VER}/bin${PATH:+:${PATH}}
            export LD_LIBRARY_PATH=/usr/local/${CUDA_VER}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        fi
        ```
        * Install required Python modules
        ```
        pip install torch torchvision torchaudio torchinfo
        ```