# 环境安装
## 环境准备
1. 环境依赖
- python==3.9.16
- PyTorch==1.13.1+cu116
- CUDA==11.6
2. 系统环境变量配置
```
export PYTHONPATH=$PWD:$PYTHONPATH
```
## 环境安装
拉取仓库
```shell
git clone ssh://git@gitlab.pjlab.org.cn:1122/CityLab/LandmarkSystem.git --recursive
```
提供基于conda的构建方式：

创建环境
```shell
conda create --name LandmarkSystem -y python=3.9.16
conda activate LandmarkSystem
```
安装PyTorch和CUDA
```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
```
安装环境依赖包：
```shell
pip install -r requirements.txt
```

安装submodules
（提示：如果您使用的是slurm集群，则须使用srun命令将下列安装指令分配到CUDA环境上执行。）
```shell
pip install landmark/ops/simple-knn/
pip install landmark/ops/opt_gaussian_rasterization/
pip install landmark/ops/fused_anchor_decoder/
```

可选项：

- 在instant-NGP和NeRFacto算法中，需要用到`tiny-cuda-nn`，可运行以下命令安装：
```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/nerfstudio-project/nerfacc.git
```

- 编译自定义Kernel

如果您想使用融合Kernel带来的推理优化，那么首先要编译和安装它们。我们的Kernel依赖cutlass，因此需要按下方说明先将cutlass作为子模块克隆下来。

注意：带有channel_last优化的gridnerf应该先编译内核。

```shell
git submodule update --init --recursive
```
然后，按照下方说明安装我们的自定义Kernel

```shell
pip install ninja --user
cd landmark/ops/
sh clean_module.sh
python setup.py install --user
```
(如果您没有指定CUTLASS_DIR，将使用子模块cutlass的默认路径。)
