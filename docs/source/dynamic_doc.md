# 动态加载策略
动态加载是为了满足在完整面积不断增大时，单张 GPU 显存无法装下完整模型参数，从而设计的一种显存换入换出策略。其利用了空间局部性，根据当前相机的坐标，将其坐标周围一定范围内的模型参数从 CPU 加载到 GPU，在迭代过程中有效地在GPU和CPU/磁盘之间加载和卸载局部参数。并通过限制相机高度和俯仰角来限制相机的视野范围，使得其只能看到已加载模型参数范围内渲染出的图像。

当相机移动时，根据一定的启发式策略预取下一范围内的模型参数，使得当相机移动到下一范围内时，可以看到预取出来的模型参数，使得其加载效果和将完整模型参数加载到 GPU 上效果一致。

动态加载渲染的先决条件都要求加载的数据是可分的，即可以按照并行策略章节的方法进行划分。

## 渲染
动态加载的渲染目前支持了3个NeRF算法，训练用于渲染的checkpoint时，唯一需要用户指定的是config文件中的`plane_division`参数。

如果需要启动动态加载渲染，则需在`offload_config`这个参数中指定和动态加载的块数划分的相关信息。下面举一个NeRF类算法的例子。

### NeRF类算法
NeRF类算法的动态加载渲染需要有前置条件，即渲染时使用的checkpoint需要在训练时是使用并行分块策略训练得到的，详见并行分块策略章节。

#### 以instant-NGP算法为例
假设在训练时使用的是分块策略为`[6, 6]`。这意味着总共被分为了36块，场景被划分为6行6列。

instant-NGP的单卡动态加载渲染启动命令为：
```shell
pytest -vxs tests/test_inference_framework/test_single_device/test_torch_inference.py::TestTorchInferenceOffload::test_offload_inference_instantNGP_no_parallel
```
需要将使用上方pytest命令所运行的函数中的instant_render_config_path里的config文件路径中的ckpt更换为您希望渲染的checkpoint路径。

假设您希望渲染的checkpoint是按照上述假设的`[6, 6]`划分的场景。那么只需要在`inference_config`中的`offload_config`里指定`local_plane_split`，这里假如我们指定其为`[3, 3]`，那么就意味着每次只加载场景中的3x3=9块场景大小的范围，以此为基础大小实现动态加载渲染。

值得注意的是，如果您更换为了自己的checkpoint，则需要将使用上方pytest命令所运行的函数最后的两个assert断言注释掉或者更改为您训练后得到的psnr值，以防止assert报错。

如果您需要存储渲染后的图片，可以将`test_inference.py`函数中的`self.save_png`变量设置为`True`即可。

最后需要注意的是，为了让动态加载的加载切换更加平滑，我们建议动态加载持有的块数划分不小于`[3, 3]`，获得更好的动态加载体验。
