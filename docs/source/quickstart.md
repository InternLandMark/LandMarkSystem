# 使用说明
## 安装
请参考[安装文档](./install.md)进行安装。
## 数据准备
### 1. 当前已经提供的数据集
#### [MatrixCity](https://city-super.github.io/matrixcity/) 数据集
##### 数据集准备

完全支持[MatrixCity](https://city-super.github.io/matrixcity/)数据集。推荐从 [OpenXLab](https://openxlab.org.cn/datasets/bdaibdai/MatrixCity) 或 [BaiduNetDisk](https://pan.baidu.com/share/init?surl=87P0e5p1hz9t5mgdJXjL1g)（密码：hqnn）下载数据集。
需要下载并按照原始目录结构组织以下文件：

```
MatrixCity/small_city/aerial/train/block_1.tar
MatrixCity/small_city/aerial/train/block_2.tar
MatrixCity/small_city/aerial/test/block_1_test.tar
MatrixCity/small_city/aerial/test/block_2_test.tar
MatrixCity/small_city/aerial/pose/block_A/
```

下载后，需要使用 `tar -xf [tar_filename]` 命令解压这些tar文件。

最后，在配置文件中需要正确设置`DatasetConfig`中的`datadir`如下所示：

(注意：如果使用高斯类算法，则必须显式指定`dataset_type`参数为`"gaussian"`,nerf类算法则不需要显式指定。`dataset_name`参数缺省值为`"city"`，可选择的范围为：["city", "matrixcity", "blender", "Colmap"]，指定后，将按照对应的数据处理逻辑进行自动处理。)
```python
dataset_config = DatasetConfig(
    datadir = YOUR_MATRIXCITY_FOLDER_PATH/small_city/aerial/pose,
    dataset_type = "gaussian",
    dataset_name = "city",
    ...
)
```
其余配置按照仓库中的示例配置文件中内容填写即可。

### 2. 如何添加新的数据集

按照以下的方式修改你的数据集：

- your_dataset/
    - images/
        - image_0.png
        - image_1.png
        - image_2.png
        - ...
    - transforms_train.json
    - transforms_test.json

`images/`文件夹包含了所有的训练和测试数据集。`transforms_xxx.json`支持多焦和单焦距格式的相机位姿。
```
### single focal example ###
{
    "camera_model": "SIMPLE_PINHOLE",
    "fl_x": 427,
    "fl_y": 427,
    "w": 547,
    "h": 365,
    "frames": [
        {
            "file_path": "./images/image_0.png",
            "transform_matrix": []
        }
    ]
}

### multi focal example ###
{
    "camera_model": "SIMPLE_PINHOLE",
    "frames": [
        {
            "fl_x": 1116,
            "fl_y": 1116,
            "w": 1420,
            "h": 1065,
            "file_path": "./images/image_0.png",
            "transform_matrix": []
        }
    ]
}
```
使用[COLMAP](https://colmap.github.io/)提取姿态和稀疏点云模型。然后使用以下命令传输位姿数据:
```shell
python app/tools/colmap2nerf.py --recon_dir data/your_dataset/sparse/0 --output_dir data/your_dataset
```
`transforms_train.Json`和`transforms_test.Json`文件将在`your_dataset/`文件夹中生成，支持单焦点。

## 启动训练
介绍下面几个算法启动训练方式，请注意将以下命令中的`config`参数对应的文件里的`datadir`变量指定为您下载的数据集的储存路径。
### gridnerf
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality.py
```

2. DDP training
- parallel degree: 2
```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_ddp_train.py
```

3. Branch Parallel training
- 2x2 division
```shell
python -m torch.distributed.launch --nproc_per_node 4 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_BranchParallel.py
```

4. Channel Parallel training

- parallel degree: 2

```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_ChannelParallel.py
```
### Nerfacto
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/nerfacto/nerfacto_trainer.py --config benchmarks/nerf/nerfacto/confs/matrixcity_2block_huge_debug.py
```

2. Branch Parallel training

- 2x1 division

```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/nerfacto/nerfacto_trainer.py --config benchmarks/nerf/nerfacto/confs/matrixcity_2block_BranchParallel.py
```
3. DDP training
- parallel degree: 2
```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/nerfacto/nerfacto_trainer.py --config benchmarks/nerf/nerfacto/confs/matrixcity_2block_lowquality_ddp_train.py
```
### Instant NGP
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/instant_ngp/instant_ngp_trainer.py --config benchmarks/nerf/instant_ngp/confs/matrixcity_2block_plconfig.py
```

2. Branch Parallel training

- 2x2 division

```shell
python -m torch.distributed.launch --nproc_per_node 4 --use_env benchmarks/nerf/instant_ngp/instant_ngp_trainer.py --config benchmarks/nerf/instant_ngp/confs/matrixcity_2block_BranchParallel.py
```

3. DDP training
- parallel degree: 2
```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/instant_ngp/instant_ngp_trainer.py --config benchmarks/nerf/instant_ngp/confs/matrixcity_2block_plconfig_ddp.py
```
### vanilla gaussian
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/origin_gs/gs_trainer.py --config benchmarks/nerf/origin_gs/confs/matrixcity_2block_train.py
```
### scaffold gaussian
Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/scaffold_gs/gs_trainer.py --config benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_train.py
```
### octree gaussian
Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/octree_gs/gs_trainer.py --config benchmarks/nerf/octree_gs/confs/matrixcity_2block_train.py
```

## 启动渲染
### 单卡渲染
在渲染的测试函数中，已经提前写好了配置路径，只需要运行即可。
#### gridnerf
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_gridnerf_no_parallel
```
#### Nerfacto
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_nerfacto_no_parallel
```
#### Instant NGP
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_instantNGP_no_parallel
```
#### vanilla gaussian
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_origin_gs_no_parallel
```
#### scaffold gaussian
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_scaffold_gs_no_parallel
```
#### octree gaussian
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_octree_gs_no_parallel
```
### 多卡渲染
为了多卡运行渲染，需要将执行的测试用例放在`.py`文件中，使用`torch.distributed.launch`命令运行，示例如下：
以InstantNGP算法为例，首先完成文件内容如下：
```python
import pytest

if __name__ == "__main__":
    pytest.main(
        [
            "-v",
            "-x",
            "-s",
            "tests/test_inference_framework/test_multi_device/test_torch_parallel_inference.py::TestTorchInferenceDP::test_instantNGP_torch_inference_dp"
        ]
    )
```
然后，命令行执行下方命令运行：
```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env test_inference_framework/test_multi_device/test_inference_dp.py
```
这里的`--nproc_per_node`参数指定了使用的GPU数量，`--use_env`参数指定了使用的环境变量，为`torch.distributed.launch`命令的参数。
其余的算法多卡渲染也是类似的操作方式。
