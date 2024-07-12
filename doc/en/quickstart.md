# User Guide
## Installation
Please refer to the [installation document](./install.md) for installation instructions.

## Data Preparation
### 1. Currently Available Datasets
#### [MatrixCity](https://city-super.github.io/matrixcity/) Dataset
##### Dataset Preparation

The [MatrixCity](https://city-super.github.io/matrixcity/) dataset is fully supported. It is recommended to download the dataset from [OpenXLab](https://openxlab.org.cn) or [BaiduNetDisk](https://pan.baidu.com) (password: hqnn).
You need to download and organize the following files according to the original directory structure:

```
MatrixCity/small_city/aerial/train/block_1.tar
MatrixCity/small_city/aerial/train/block_2.tar
MatrixCity/small_city/aerial/test/block_1_test.tar
MatrixCity/small_city/aerial/test/block_2_test.tar
MatrixCity/small_city/aerial/pose/block_A/
```

After downloading, you need to use the `tar -xf [tar_filename]` command to extract these tar files.

Finally, you need to correctly set the `datadir` in `DatasetConfig` in the configuration file as shown below:

(Note: If using Gaussian-type algorithms, you must explicitly specify the `dataset_type` parameter as `"gaussian"`. For NeRF-type algorithms, this is not required. The default value for the `dataset_name` parameter is `"city"`, with optional values being: ["city", "matrixcity", "blender", "Colmap"]. Once specified, the corresponding data processing logic will be automatically applied.)

```python
dataset_config = DatasetConfig(
    datadir = YOUR_MATRIXCITY_FOLDER_PATH/small_city/aerial/pose,
    dataset_type = "gaussian",
    dataset_name = "city",
    ...
)
```
The rest of the configuration can be filled in according to the example configuration file in the repository.

### 2. How to Add New Datasets

Modify your dataset according to the following structure:

- your_dataset/
    - images/
        - image_0.png
        - image_1.png
        - image_2.png
        - ...
    - transforms_train.json
    - transforms_test.json

The `images/` folder contains all training and testing datasets. `transforms_xxx.json` supports both multi-focal and single-focal camera pose formats.

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
Use [COLMAP](https://colmap.github.io/) to extract poses and sparse point cloud models. Then use the following command to transfer pose data:
```shell
python app/tools/colmap2nerf.py --recon_dir data/your_dataset/sparse/0 --output_dir data/your_dataset
```
The `transforms_train.json` and `transforms_test.json` files will be generated in the `your_dataset/` folder, supporting single focal points.

## Start Training
The following introduces the training startup methods for several algorithms. Please note that in the `config` parameter files of the following commands, the `datadir` variable should be specified as the storage path of the dataset you downloaded.

### gridnerf
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality.py
```

2. DDP training

```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_ddp_train.py
```

3. Branch Parallel training

```shell
python -m torch.distributed.launch --nproc_per_node 4 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_BranchParallel.py
```

4. Channel Parallel training

- parallel degree: 2

```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/gridnerf/gridnerf_trainer.py --config benchmarks/nerf/gridnerf/confs/matrixcity_2block_lowquality_ChannelParallel.py
```

### nerfacto
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/nerfacto/nerfacto_trainer.py --config benchmarks/nerf/nerfacto/confs/matrixcity_2block_huge_debug.py
```

2. Branch Parallel training

- 2x1 division

```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env benchmarks/nerf/nerfacto/nerfacto_trainer.py --config benchmarks/nerf/nerfacto/confs/matrixcity_2block_BranchParallel.py
```

### instant ngp
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/instant_ngp/instant_ngp_trainer.py --config benchmarks/nerf/instant_ngp/confs/matrixcity_2block_plconfig.py
```

2. Branch Parallel training

- 2x2 division

```shell
python -m torch.distributed.launch --nproc_per_node 4 --use_env benchmarks/nerf/instant_ngp/instant_ngp_trainer.py --config benchmarks/nerf/instant_ngp/confs/matrixcity_2block_BranchParallel.py
```

### vanilla gaussian
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/origin_gs/gs_trainer.py --config benchmarks/nerf/origin_gs/confs/matrixcity_2block_train.py
```
### scaffold gaussian
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/scaffold_gs/gs_trainer.py --config benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_train.py
```
### octree gaussian
1. Single GPU training

```shell
python -m torch.distributed.launch --nproc_per_node 1 --use_env benchmarks/nerf/octree_gs/gs_trainer.py --config benchmarks/nerf/octree_gs/confs/matrixcity_2block_train.py
```
## Start Rendering
### Single GPU Rendering
In the rendering test functions, the configuration paths have already been pre-written, so you only need to run them.

#### gridnerf
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_gridnerf_no_parallel
```

#### nerfacto
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceNoParallel::test_inference_nerfacto_no_parallel
```

#### instant ngp
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

### Multi-GPU Rendering
To run rendering on multiple GPUs, you need to put the test case to be executed in a `.py` file and run it using the `torch.distributed.launch` command. Here's an example:

Taking the instantNGP algorithm as an example, first complete the file content as follows:
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
Then, execute the following command in the command line to run:
```shell
python -m torch.distributed.launch --nproc_per_node 2 --use_env test_inference_framework/test_multi_device/test_inference_dp.py
```
Here, the `--nproc_per_node` parameter specifies the number of GPUs to use, and the `--use_env` parameter specifies the environment variables to use.
The multi-GPU rendering for other algorithms follows a similar operation method.
