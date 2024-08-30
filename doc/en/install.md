# Environment Installation

## Environment Preparation

1. Environment Dependencies
- python==3.9.16
- PyTorch==1.13.1+cu116
- CUDA==11.6

2. System Environment Variable Configuration
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Installation Steps

Clone the repository
```shell
git clone git@github.com:InternLandMark/LandMarkSystem.git --recursive
```

Conda-based setup method provided

Create environment
```shell
conda create --name LandmarkSystem -y python=3.9.16
conda activate LandmarkSystem
```

Install PyTorch and CUDA
```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
```

Install environment dependencies:
```shell
pip install -r requirements.txt
```

Install submodules
(Note: If you're using a slurm cluster, use the srun command to allocate the following installation instructions to the CUDA environment for execution)
```shell
pip install landmark/ops/simple-knn/
pip install landmark/ops/opt_gaussian_rasterization/
pip install landmark/ops/fused_anchor_decoder/
```

Optional:

- For instant-NGP and NeRFacto algorithms, `tiny-cuda-nn` is required. Run the following commands to install:
```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/nerfstudio-project/nerfacc.git
```

- Compile Custom Kernels

If you want to use the inference optimization brought by fused kernels, you need to compile and install them first. Our kernels depend on cutlass, so you need to clone cutlass as a submodule according to the instructions below.

Note: gridnerf with channel_last optimization should compile the kernel first.

```shell
git submodule update --init --recursive
```

Then, install our custom kernels according to the instructions below

```shell
pip install ninja --user
cd landmark/ops/
. clean_module.sh
python setup.py install --user
```
(If you don't specify CUTLASS_DIR, the default path of the cutlass submodule will be used.)
