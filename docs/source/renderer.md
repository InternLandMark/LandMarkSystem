# 渲染构建
构造渲染引擎主要由下面4个部分构成，关于它们的详细描述可参考[渲染API](./rendering_api.rst)：

- 定义算法的类：算法的定义类，用于实例化算法模型

- 定义算法推理逻辑的类（可选）：对模型推理逻辑的封装类，主要用于推理的适配，管理前处理和后处理逻辑

- 模型配置文件：包含模型的配置参数，模型checkpoint等内容

- 推理配置文件：定义引擎推理优化的配置

这里我们同样将基于 GridNeRF 和 OctreeGS 展开介绍如何使用我们的渲染引擎。

## GridNeRF
### ModelClass

模型定义请参考[训练构建](./trainer.md)部分。

### InferenceClass

对于 GridNeRF 的推理模型定义可参考 `tests/utils.py` 中的定义，首先先需要继承基类 `InferenceModule`，然后实现初始化方法。

```

class InferenceGridNerfModule(InferenceModule):
    """
    Gridnerf inference module
    """

    def __init__(self, *args, **kwargs):
        self.render_1080p = True
        model = GridNeRF(*args, **kwargs)
        super().__init__(model=model)

```

在 `__init__` 方法中，我们需要指定输入参数为 `*args, **kwargs` 来避免重复定义 GridNeRF 的输入，然后将实例化后的模型 model 传入给基类即可。 除此之外，我们还可以定义一些复杂的控制逻辑，比如在这里就定义了 render_1080p 属性。

之后，我们再实现 `preprocess` 、 `forward` 和 `postprocess` 的逻辑。

```

class InferenceGridNerfModule(InferenceModule):
    """
    Gridnerf inference module
    """

    def __init__(self, *args, **kwargs):
        self.render_1080p = True
        model = GridNeRF(*args, **kwargs)
        super().__init__(model=model)


    def preprocess(self, pose, chunk_size, H, W, app_code, edit_mode):
        assert edit_mode is not None
        # self.model.edit_model(edit_mode)

        if self.render_1080p:
            self.H, self.W = 1080, 1920
        else:
            self.H, self.W = H, W

        focal = 2317.6449482429634 if self.render_1080p else pose[-1, -1]
        rays = self.generate_rays(pose, H, W, focal)
        args = (rays, chunk_size, app_code)
        kwargs = {}
        return args, kwargs

    def forward(self, rays, chunk_size, app_code):
        N_samples = -1
        idxs = torch.zeros_like(rays[:, 0], dtype=torch.long, device=rays.device)  # TODO need check
        all_ret = self.model.render_all_rays(rays, chunk_size, N_samples, idxs, app_code)
        return all_ret["rgb_map"]

    def postprocess(self, result):
        result = result.clamp(0.0, 1.0)
        result = result.reshape(self.H, self.W, 3) * 255
        result = torch.cat([result, torch.ones((self.H, self.W, 1), device=result.device) * 255], dim=-1)
        return result

```

在 `preprocess` 中，我们将输入的 pose 转换成了模型需要的 rays 。考虑到 model 的 `forward` 仅支持部分 rays，我们通过在 `InferenceGridNerfModule` 中对 `forward` 进行封装，达到支持完整的 pose 这一目的。最后，在 `postprocess` 中，根据需求对输出进行适配。

### ModelConfig
渲染时用于构造模型的配置文件由三个部分组成，分别是 `ModelConfig` ， `RenderConfig` 和 `DatasetConfig` ，这些参数都可以基于训练配置所直接获取。

如`benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_render.py`所示

```
from benchmarks.nerf.gridnerf.confs.matrixcity_2block_multi import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
ckpt="/mnt/hwfile/landmark/checkpoint/landmark_sys/gridnerf/matrix_city_blockA_multi/state_dict.th",
kwargs="/mnt/hwfile/landmark/checkpoint/landmark_sys/gridnerf/matrix_city_blockA_multi/kwargs.th",
).from_train_config(train_config)

model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]

```

在 `RenderConfig` 中，需要指定下面的字段：

- ckpt: 指定存储模型状态参数 state dict 的文件路径（必须）
- kwargs: 指定存储模型初始化参数的文件路径（可选，取决于模型定义时是否指定 kwargs）

针对 GridNeRF 算法，我们在模型层实现了光线采样点的优化，可以通过修改 `ModelConfig` 相应的配置控制，这里我们将更新模型参数  `model_config.sampling_opt = True` 来启动采样点的优化。

### InferenceConfig

推理配置文件主要用于控制推理的行为，对于 GridNeRF 来说，我们支持串行渲染，数据并行渲染，模型并行渲染以及动态加载渲染，对应的配置文件也有所不同。

串行渲染配置
```
inference_config = dict(
    runtime="Torch",
    kernel_fusion=False,
)
```

并行渲染配置
```
inference_config = dict(
    parallel_config=dict(
        tp_size=2,
    ),
    runtime="Torch",
    kernel_fusion=False,
)
```
并行主要通过追加 `parallel_config` 进行实现， `tp_size` 用于控制并行度，存在以下公式: `world_size = tp_size * dp_size`。当 `world_size` 为 8 时，上面的控制表示2路模型并行，4路数据并行。

动态加载渲染配置
```
inference_config = dict(
    runtime="Torch",
    kernel_fusion=False,
    offload_config=dict(
        local_plane_split=[1, 1],
    ),
)
```
动态加载主要通过追加 `offload_config` 进行实现， `local_plane_split` 用于控制每次加载到显存的块数。

**注：对于 GridNeRF 算法来说，开启动态加载需要配合 Branch Parallel 训练的模型进行使用**

动态加载配合数据并行
```
inference_config = dict(
    parallel_config=dict(
        tp_size=1,
    ),
    runtime="Torch",
    kernel_fusion=False,
    offload_config=dict(
        local_plane_split=[1, 1],
    ),
)
```
通过指定 `parallel_config` 和 `offload_config` ，可以开启数据并行的动态加载（注: 对于 GridNeRF 来说仅支持数据并行加上动态加载，故 `tp_size` 必须指定为1）。数据并行的路数仅和启动的 `world_size` 相关。

### 开始渲染
在准备好前面的类和配置之后，可以正式使用引擎开始渲染：

```
import os

from benchmarks.nerf.gridnerf.gridnerf import GridNeRF
from landmark import init_inference
from landmark.nerf_components.configs.config_parser import BaseConfig
from landmark.nerf_components.data import DatasetManager
from landmark.nerf_components.utils.image_utils import psnr as cal_gs_psnr
from landmark.utils.env import EnvSetting
from tests.utils import InferenceGridNerfModule

gridnerf_render_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_render.py",
)
print(f"{gridnerf_render_config_path=}", flush=True)
model_config = BaseConfig.from_file(gridnerf_render_config_path)
inference_config = dict(
    parallel_config=dict(
        tp_size=1,
    ),
    runtime="Torch",
    kernel_fusion=False,
    offload_config=dict(
        local_plane_split=[1, 1],
        use_nccl=True,
    ),
)
engine = init_inference(GridNeRF, InferenceGridNerfModule, model_config, inference_config)

dataset_mgr = DatasetManager(model_config)
dataset = dataset_mgr.test_dataset
W, H = dataset.img_wh
chunk_size = model_config.render_batch_size
edit_mode = {"idx", 0}
app_code = 0
for idx in range(dataset_mgr.test_data_size):
    if EnvSetting.RANK == 0:
        gt_rgb = dataset.all_rgbs[idx].view(H, W, 3)
        forward_args = (dataset.poses[idx], chunk_size, H, W, app_code, edit_mode)
        nerf_out = engine(*forward_args)
        nerf_out = nerf_out[..., :-1]
        rgb_map = nerf_out / 255.0
        rgb_map = rgb_map.cpu()
        psnr = cal_gs_psnr(rgb_map, gt_rgb).mean().item()
        print(f"{psnr=}")
    else:
        nerf_out = engine()
```

将上述代码保存为 `test_gridnerf_render.py` 文件，然后使用下面渲染指令启动渲染即可 `torchrun --nproc_per_node 2 ./test_gridnerf_render.py`

## OctreeGS
### ModelClass
模型定义请参考[训练构建](./trainer.md)部分。

### InferenceClass

对于 OctreeGS 的推理模型定义可参考 `tests/utils.py` 中的定义，首先先需要继承基类 `InferenceModule`，然后实现初始化方法。

```
class InferenceOctreeGSModule(InferenceModule):
    """
    OctreeGS inference module
    """

    def __init__(self, *args, **kwargs):
        model = OctreeGS(*args, **kwargs)
        super().__init__(model=model)

    def preprocess(self, viewpoint_cam, scaling_modifier, retain_grad, render_exclude_filter):
        viewpoint_cam.image = None
        viewpoint_cam.c2w = None
        viewpoint_cam.R = None
        viewpoint_cam.T = None
        viewpoint_cam.trans = None
        viewpoint_cam.image_name = None
        viewpoint_cam.image_path = None
        ape_code = -1
        args = (viewpoint_cam, scaling_modifier, retain_grad, render_exclude_filter, ape_code)

        kwargs = {}
        return args, kwargs

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    def postprocess(self, out):
        return out
```

对于 `GS` 类算法来说，输入为 `Camera` ，考虑到数据并行的情况下，会首先将 `preprocess` 的输出从 rank0 广播到其他 rank，为了减少通信的数据量，需要先将 `Camera` 在 `preprocess` 进行裁剪。然后根据模型的 `forward` 构造其他相关的参数即可。

### ModelConfig
渲染时用于构造模型的配置文件由三个部分组成，分别是 `ModelConfig` ， `RenderConfig` 和 `DatasetConfig` ，这些参数都可以基于训练配置所直接获取。

如`benchmarks/nerf/octree_gs/confs/matrixcity_2block_render.py`所示

```
from benchmarks.nerf.octree_gs.confs.matrixcity_2block_train import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="/mnt/hwfile/landmark/checkpoint/landmark_sys/octree_gs/OctreeGS_matrix_city_blockA/point_cloud.ply",
    ckpt_type="full",
    batch_size=1,
    N_vis=20,
).from_train_config(train_config)

model_config.enable_lod = True

config = [render_config, dataset_config, model_config]

```

在 `RenderConfig` 中，需要指定下面的字段：

- ckpt: 指定存储模型状态参数 state dict 的文件路径（必须）

针对 OctreeGS 算法，我们在模型层实现了 `LOD` 的优化，可以通过修改 `ModelConfig` 相应的配置控制，这里我们将更新模型参数  `model_config.enable_lod = True` 来启动采样点的优化。

不同于 GridNeRF ，这里没有 kwargs 参数，这是因为 OctreeGS 并没有在模型定义时指定 kwargs 的内容，因此在训练过程中也不会保存 kwargs 的初始化参数。

对于 Octree GS 来说，训练后除了生成存储 `GassusianEncoding` 参数的 `point_cloud.ply` 之外，还会保存其他的模型参数 `state_dict.th` ，它与 `point_cloud.ply` 位于相同目录，在这里无需额外指定，只需要保证存储位置位于同一目录下，引擎侧在读取参数时会将两者进行合并。

### InferenceConfig

推理配置文件主要用于控制推理的行为，对于 OctreeGS 来说，我们支持串行渲染，数据并行渲染以及动态加载渲染，对应的配置文件也有所不同。

串行渲染配置
```
inference_config = dict(
    runtime="Torch",
    kernel_fusion=False,
)
```

为了进一步提升 OctreeGS 的性能，我们还对相关组件做了 kernel 层级的优化，只需要按如下方式指定即可启动

```
inference_config = dict(
    runtime="Kernel",
    kernel_fusion=True,
)
```

### 开始渲染
在准备好前面的类和配置之后，可以正式使用引擎开始渲染了：
```python
import os

from benchmarks.nerf.octree_gs.gs_model import OctreeGS
from landmark import init_inference
from landmark.nerf_components.configs.config_parser import BaseConfig
from landmark.nerf_components.data import DatasetManager
from landmark.nerf_components.utils.image_utils import psnr as cal_gs_psnr
from landmark.utils.env import EnvSetting
from tests.utils import InferenceOctreeGSModule

octree_gs_render_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks/nerf/octree_gs/confs/matrixcity_2block_render.py",
)
model_config = BaseConfig.from_file(octree_gs_render_config_path)
inference_config = dict(
    runtime="Kernel",
    kernel_fusion=True,
)
engine = init_inference(OctreeGS, InferenceOctreeGSModule, model_config, inference_config)

dataset_mgr = DatasetManager(model_config)
for idx in range(dataset_mgr.test_data_size):
    if EnvSetting.RANK == 0:
        viewpoint_cam, gt_rgb = dataset_mgr.get_camera("test")
        forward_args = (viewpoint_cam, 1.0, None, None)
        gs_out = engine(*forward_args)
        rgb_map = gs_out.clamp(0.0, 1.0).cpu()
        psnr = cal_gs_psnr(rgb_map, gt_rgb).mean().item()
        print(f"{psnr=}", flush=True)
    else:
        gs_out = engine()

```

将上述代码保存为 `test_octree_render.py` 文件，然后使用下面渲染指令启动渲染即可 `torchrun --nproc_per_node 1 ./test_octree_render.py`
