# 训练构建

本节将以LandmarkSystem中NeRF类算法GridNeRF以及3DGS类算法Octree-GS为例，对其模型的组件化构建、训练的重要参数配置以及并行化的使用方法进行说明。

## 模型构建

### GridNeRF

利用LandmarkSystem所提供的算法组件，我们可以构造GridNeRF模型的基础结构，并定义模型的forward计算流程。
构造GridNeRF基础结构所需的组件有以下几个，可以从`landmark.nerf_components.model_components`中取得：

- TensorVMEncoding # GridNeRF模型网格分支，将场景特征编码为三维矩阵的VM分解形式
- NeRF # GridNeRF模型NeRF分支
- AlphaGridMask # 记录场景中对渲染有贡献区域的三维网格掩码
- AppearanceEmbedding # 用于对外观特征进行嵌入，学习不一致的场景外观
- MLPDecoder # 将密度与颜色的特征向量解码为输出值
- VolumeRenderer # 将光线上的所有采样点以体积渲染方式积分为对应像素

对于NeRF类算法而言，采样器是除模型基础结构外的主要模块，在GridNeRF中，主要使用以下两个采样器

- PDFSampler 逆变换采样器
- UniformSampler 均匀采样器

所有采样器组件均可从`landmark.nerf_components.ray_samplers`中取得

当准备好上述组件后，我们可以继承`landmark.nerf_components.model.BaseNeRF`并定义GridNeRF。具体地，首先需要定义模型初始化阶段的三个抽象方法：

- init_field_components() # 初始化模型的field组件，包括`TensorVMEncoding`、`NeRF`、`AppearanceEmbedding`、`RGBDecoder`
- init_sampler() # 初始化模型的采样器
- init_renderer() # 初始化体积渲染器`VolumeRenderer`

接着，我们需要构造GridNeRF的forward计算流程，主要包含`空间采样`、`特征计算`、`特征解码`、`积分渲染`四个阶段，并遵循如下输入输出要求：

- 输入：rays_chunk、N_samples # 光线集合以及单条光线上的采样数
- 输出：outputs # 所有光线积分后的像素颜色结果，包括RGB图像、NeRF端图像以及深度图等

模型的构造的具体细节可参考`benchmarks.nerf.gridnerf.GridNeRF`，包含了forward计算流程的并行化实现

### Octree-GS

利用LandmarkSystem所提供的算法组件，我们可以构造Octree-GS模型的基础结构，并定义模型的forward计算流程。
构造Octree-GS基础结构所需的组件有以下几个，可以从`landmark.nerf_components.model_components`中取得

- OctreeGSEncoding # OctreeGS形式的场景表征，将特征编码为八叉树结构的特征锚点
- AppearanceEmbedding # 用于对外观特征进行嵌入，学习不一致的场景外观
- AnchorDecoder # 用于将锚定的特征向量解码为实际用于渲染的高斯球参数

对于3DGS类算法而言，光栅化器是除模型基础结构外的主要模块，在OctreeGS中，主要使用以下光栅化器

- GaussianRasterizer # 通用的高斯球光栅化渲染器
- GaussianRasterizationSettings # 用于对光栅化器渲染参数进行配置

所有光栅化器组件均可从`opt_diff_gaussian_rasterization`中取得

当准备好上述组件后，我们可以继承`landmark.nerf_components.model.BaseGaussian`并定义Octree-GS。具体地，首先需要定义模型初始化阶段的两个抽象方法：

- init_field_components() # 初始化模型的field组件，包括`OctreeGSEncoding`、`AnchorDecoder`
- init_renderer() # 使用`GaussianRasterizationSettings`对光栅化器的默认参数进行配置

接着，我们需要构造Octree-GS的forward计算流程，主要包含`anchor过滤`、`anchor解码`、`光栅化渲染`、三个阶段，并遵循如下输入输出要求：

- 输入：viewpoint_camera、scaling_modifier、retain_grad、 ape_code # `Camera`类型的相机参数、高斯球尺寸缩放比、梯度保留标志位（用于训练）、appearance embedding所使用的外观嵌入编号
- 输出：rendered_image, scaling # RGB图像、高斯球尺寸

对于3DGS类算法而言，稠密化（densification）方法是除了模型在训练过程中重要的离线计算操作，被包含于相应的gaussian_encoding组件中。在OctreeGS中，其实现主要包含了两个方法：

- gaussian_encoding.training_statis() # 在到达指定的稠密化iter数前，对稠密化所需的参数进行更新
- gaussian_encoding.adjust_anchor() # 在到达指定的稠密化iter数时，对anchor执行稠密化操作

模型的构造的具体细节可参考`benchmarks.nerf.octree_gs.OctreeGS`以及`OctreeGSEncoding`，包含了forward流程以及稠密化的具体实现

## 参数解析

在训练配置文件中，需要包含三个config类型，包括`ModelConfig`、`DatasetConfig`、`TrainConfig`。其中`ModelConfig`需要根据实际使用的模型进行选择，如`GridNeRFConfig`或`OctreeGSConfig`；`DatasetConfig`和`TrainConfig`是配置数据集参数与训练参数的通用config类型。

### GridNeRF

在GridNeRF的相关训练参数中，一些重要的模型、数据集以及训练配置示例如下

#### GridNeRFConfig

- resMode=[1, 2, 4] # 特征网格金字塔的分辨率模式，此处设置了1,2,4三级并逐级倍增
- encode_app=False # 是否在训练中开启appearance embedding
- nonlinear_density=True # 在计算采样点密度时是否采用非线性激活方式
- add_nerf=40000 # 训练中开始加入NeRF Branch的iter数
- N_voxel_init=128**3 # 初始网格体素数量，此处定义了一个分辨率为128的网格。在多分辨率模式下，定义为第一级网格的体素数量
- N_voxel_final=300**3 # 最终网格体素数量，此处定义了一个分辨率为300的网格。在多分辨率模式下，定义为第一级网格的体素数量
- upsamp_list=[2000, 3000, 4000, 5500, 7000] # 训练中进行网格上采样的iter数列表，网格将从初始分辨率逐步上采样提升到最终分辨率
- update_AlphaMask_list=[2000, 4000, 8000, 10000] # 训练中对AlphaMask网格进行更新的iter数列表
- appearance_embedding_size=1063 # 开启appearance embedding时，指定embedding总数。该值应当等同于训练集图像的数量

#### DatasetConfig

- dataset_type="nerf" # dataloader所要适配的模型类型
- dataset_name="matrixcity" # dataloader所要适配的数据集格式
- datadir=MatrixCity/small_city/aerial/pose/block_A" # 数据集路径
- lb=[-1.0378e01, -7.4430e00, -1.1921e-07] # 数据集中场景的AABB下界
- ub=[1.2078e00, 2.4078e00, 9.0000e-01] # 数据集中场景的AABB上界

#### TrainConfig

- model_name="GridNeRF" # 训练对应的模型名称
- expname="matrix_city_blockA_multi" # 训练对应的实验名称
- basedir="./log" # 存放训练输出的基础路径
- batch_size=8192 # 训练输入的采样光线数量
- vis_every=5000 # 模型进行图像渲染的训练步数间隔
- n_iters=50000 # 模型训练过程的总训练步数
- N_vis=5 # 渲染时在test集中的选取总数

### OctreeGS

在Octree-GS的相关训练参数中，一些重要的模型、数据集以及训练配置示例如下

#### OctreeGSConfig

- appearance_dim=0 # appearance embedding的特征维度，为0时表示不使用appearance embedding
- fork=2 # 相邻LOD层级之间的分支数，为N时将产生N^3数量的分支
- base_layer=-1 # 将部分Octree层级作为训练初始化模型时的初始层，-1时将自动选取
- visible_threshold=0.9 # anchor在所有训练视角下的可见性阈值，未达到时将删除anchor
- dist2level="round" # 距离转为层级level时所使用的方法，训练时为`round`，渲染时为`progressive`
- update_ratio=0.2 # 对anchor进行调整时的随机更新比例
- progressive=True # 渐进式训练标志位，为True时会逐步增加参与训练的层级
- dist_ratio=0.999 # 距离尺度计算时的覆盖比例，0.999意味着取最大的0.001和最小的0.001为距离极值
- levels=-1 # Octree的总层级数，也即LOD级数，为-1时将自动进行计算
- init_level=-1 # 训练起始的层级数，为-1时将自动选取其实层级
- extra_ratio=0.5 # LOD层级的渐变性过渡值
- extra_up=0.01 # anchor在训练中进行LOD层级提升的步长
- white_bkgd=False # 是否使用白色的图像背景
- resolution=1 # 训练图像的缩放比例，大于1时将进行降采样缩小
- resolution_scales=[1.0] # 训练图像多缩放尺度列表，在训练中引入同一数据集图像不同缩放的副本
- point_path=None # 训练初始化模型时所使用的点云路径

#### DatasetConfig

- dataset_type="gaussian" # dataloader所要适配的模型类型
- dataset_name="city" # dataloader所要适配的数据集格式
- datadir="MatrixCity/small_city/aerial/pose/block_A/" # 数据集路径
- preload=True # 是否使用数据集预加载，不使用时将在训练初期逐iter加载

#### TrainConfig

- model_name="OctreeGS" # 训练对应的模型名称
- expname="OctreeGS_matrix_city_blockA" # 训练对应的实验名称
- basedir="./log" # 存放训练输出的基础路径
- batch_size=1 # 训练输入的batch_size大小，1表示1个相机位姿及对应图像
- n_iters=40000 # 模型训练过程的总训练步数
- test_iterations=[10000, 20000, 30000, 40000] # 训练过程中进行渲染测试的iter数列表
- save_iterations=[10000, 20000, 30000, 40000] # 训练过程中进行保存的iter数列表
- tensorboard=True # 是否在训练过程中启用tensorboard

## 训练初始化

虽然GridNeRF和Octree-GS在模型结构和训练流程上存在差异，但均可继承`landmark.train.NeRFTrainer`并调用相关API以构造对应的模型训练Trainer

### GridNeRFTrainer

在`benchmarks.nerf.gridnerf.GridNeRFTrainer`的初始化阶段，主要涉及到以下几个API的调用

```
class GridNeRFTrainer(NeRFTrainer):
    """Trainer class for GridNeRF"""

    def __init__(self,config: BaseConfig,):
        super().__init__(config)
        self.init_train_env() # 初始化训练环境中的随机数种子与分布式
        self.scene_mgr = SceneManager(config) # 初始化SceneManager，用于在并行化训练中为采样点分配相应的block编号
        self.data_mgr = DatasetManager(config) # 初始化DatasetManager，用于管理dataloader等与数据集相关的参数及操作
        self.model = self.create_model() # 构建GridNeRF模型
        self.optimizer = self.create_optimizer() # 构建模型参数优化器
        self.check_args() # 对部分训练参数的互斥性和正确性进行检查
```

其中，`SceneManager`和`DatasetManager`作为训练组件，均可从`landmark.nerf_components`中取得；`create_model()`以及`create_optimizer()`则需要在`GridNeRFTrainer`进行重写以支持并行化的训练。

借助`landmark.nerf_components.ComponentConvertorFactory`可以将在构建模型时简便地将模型结构转换为支持model parallel模型并行训练的并行形式

```
def create_model(self):
    config = self.config
    aabb = self.data_mgr.dataset_info.aabb.to(config.device) # 取得数据集中的AABB参数
    reso_cur = n_to_reso(config.N_voxel_init, aabb)
    gridnerf = GridNeRF(  # 构造基础的GridNeRF模型
        aabb,
        reso_cur,
        device=config.device,
        near_far=near_far,
        scene_manager=self.scene_mgr,
        config=config,
    )

    if config.model_parallel:
        parallelize_convert = ComponentConvertorFactory.get_convertor("parallelize")
        gridnerf = parallelize_convert.convert(gridnerf, config) # 将基础的GridNeRF模型转换为并行形式
    return gridnerf
```

`train()`和`evaluation()`是`GridNeRFTrainer`中执行模型训练循环和验证循环的基础方法，其中`train()`执行步数由参数`n_iters`决定，每个训练步输入的数据量为`batch_size`所决定的采样光线数量；`evaluation()`执行步数由`N_vis`决定，每步输入为图像像素数相等的采样光线。


### OctreeGSTrainer

在`benchmarks.nerf.octree_gs.OctreeGSTrainer`的OctreeGSTrainer，所涉及的主要API调用与`GridNeRFTrainer`相同，但重写`create_model()`方法时，3DGS类算法与NeRF类算法存在很大的差异，示例如下
```
def create_model(self):
    config = self.config
    gaussians = OctreeGS(config)

    pcd = get_pcd(loaded_iter, config, self.aabb, self.logfolder, gaussians) # 加载点云或生成随机点云

    print("setting appearance embedding...")
    gaussians.set_appearance(self.data_mgr.train_data_size) # 设置appearance embedding参数
    points = torch.tensor(pcd.points[:: config.ratio]).float().cuda()
    unique_points = torch.unique(points, dim=0)
    print("setting LOD levels...")
    gaussians.set_level(unique_points, self.data_mgr.train_dataset.cams) # 设置LOD层级数
    print("setting progressive intervals...")
    gaussians.set_coarse_interval() # 设置Octree-GS渐进式训练中的“粗粒度”阶段步数
    print("setting initial octree...")
    pcd._replace(points=unique_points.cpu().numpy())

    gaussians.create_from_pcd(pcd, self.data_mgr.train_dataset.cameras_extent) # 使用点云初始化模型

    self.spatial_lr_scale = self.data_mgr.train_dataset.cameras_extent # 根据数据集设置空间学习率缩放
```
在3DGS类算法的`train()`中，需要在每一个训练步调用梯度回传后、`optimizer.step()`前调用模型所定义的稠密化操作`densification()`，例如`OctreeGS.densification()`，以进行模型高斯球数量的增删调整；3DGS类算法的`train()`与`evaluation()`单步过程上有很大的相似性，每个训练步与验证步的主要计算流程都是对单一的相机位姿进行渲染，差异在于`train()`中则还包含了计算loss、梯度回传、参数优化等训练相关操作

## 并行化训练

NeRF类算法支持多种并行化的训练形式，以GridNeRF为例，其支持data parallel、channel parallel以及branch parallel等并行方法。

### GridNeRF

在示例的`benchmarks.nerf.gridnerf.GridNeRF`和`GridNeRFTrainer`中，已对前述并行方法进行适配，通过调整训练配置文件参数以启用相应的并行训练方法：

- DDP=True # 启用分布式数据并行训练
- channel_parallel=True # 启用channel并行训练
- channel_parallel_size=2 # 设置channel并行维度为2
- branch_parallel=True # 启用branch并行训练
- plane_division=[4, 4] # 设置branch并行维度为4x4
- model_parallel_and_DDP=True # 启用混合并行训练

在设置并行训练参数后，可通过`torch.distributed.launch`或`torchrun`启动多卡并行的训练实验。以2并行度的`channel_parallel`为例，启动指令如下

```
python -m torch.distributed.launch --nproc_per_node 4 --use_env parallel_train_config.py
```

或

```
torchrun --nproc_per_node 4 --use_env parallel_train_config.py
```

其中，分布式中环境的world size要与所配置的总并行度保持一致
