算法组件API
============

介绍算法组件相关的api，包括基础 ``Configs``, ``BaseModels``, ``Components`` 和 ``Ray Samplers`` 。

Configs
----------
config中包含数据集配置、模型配置和训练配置，分别对应DatasetConfig、ModelConfig和TrainConfig。
定义模型配置类的时候，需要继承自 ``landmark.nerf_components.configs.ConfigClass`` 类，
并实现 ``check_args`` 函数，该函数用于检查参数是否有效。

.. autoclass:: landmark.nerf_components.configs.ConfigClass
    :members:

定义模型配置类的示例：

.. code-block:: python

    from landmark.nerf_components.configs import ConfigClass
    class OctreeGSConfig(ConfigClass):
        """Octree-GS Config"""

        feat_dim: int = 32
        view_dim: int = 3
        n_offsets: int = 10
        fork: int = 2

        use_feat_bank: bool = False
        source_path: str = ""
        model_path: str = ""
        images: str = "images"
        resolution: int = 1
        white_background: bool = False
        random_background: bool = False
        resolution_scales: List[float] = [1.0]

        ...

.. autoclass:: landmark.nerf_components.configs.DatasetConfig
    :members: check_args

.. autoclass:: landmark.nerf_components.configs.RenderConfig
    :members: from_train_config, check_args

.. autoclass:: landmark.nerf_components.configs.TrainConfig
    :members: check_args

Base Models
---------
BaseModels中包含了NeRF的基础模型BaseNeRF，Gaussian的基础模型BaseGaussian，自定义模型需要继承基础模型，并实现其中方法。
NeRF模型可以参考benchmarks/nerf/gridnerf/gridnerf.py，Gaussian模型可以参考benchmarks/nerf/origin_gs/gs_model.py。


.. autoclass:: landmark.nerf_components.model.base_model.Model
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model.base_model.BaseNeRF
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model.base_model.BaseGaussian
    :members:
    :undoc-members:

Model Components
----------------
Components中提供了常用的算法组件，建议使用这里的组件进行模型定义。

.. autoclass:: landmark.nerf_components.model_components.fields.alpha_mask.AlphaGridMask
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model_components.fields.anchor_decoder.AnchorDecoder
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model_components.fields.embedding.AppearanceEmbedding
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model_components.MLPDecoder
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model_components.NeRF
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.model_components.VolumeRenderer
    :members:
    :undoc-members:

Ray Samplers
-------------
Ray Samplers中提供了常用的光线采样方法，一般用于NeRF的模型中，可以根据需要选择使用。实现自定义采样器建议继承BaseSampler，并实现其中的方法。

.. autoclass:: landmark.nerf_components.ray_samplers.base_sampler.BaseSampler
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.ray_samplers.uniform_sampler.UniformSampler
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.ray_samplers.pdf_sampler.PDFSampler
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.ray_samplers.proposal_network_sampler.ProposalNetworkSampler
    :members:
    :undoc-members:

.. autoclass:: landmark.nerf_components.ray_samplers.volumetric_sampler.VolumetricSampler
    :members:
    :undoc-members:
