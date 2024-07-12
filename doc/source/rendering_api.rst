渲染API
============
本章将介绍渲染相关的 API。

landmark.init_inference
----------

.. autofunction:: landmark.init_inference

landmark.InferenceModule
----------

.. autoclass:: landmark.InferenceModule
    :members:

landmark.nerf_components.configs.config_parser.BaseConfig
----------

.. autoclass:: landmark.nerf_components.configs.config_parser.BaseConfig
    :members:

示例:

.. code-block:: python

    import os
    from landmark.nerf_components.configs.config_parser import BaseConfig

    octree_gs_render_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmarks/nerf/octree_gs/confs/matrixcity_2block_render_ddp.py",
    )
    model_config = BaseConfig.from_file(octree_gs_render_config_path)
    inference_config_dict = dict(
        parallel_config=dict(
            tp_size=1,
        ),
        runtime="Kernel",
        kernel_fusion=True,
        offload_config=dict(
            plane_split=[2, 1],
            local_plane_split=[1, 1],
            use_nccl=False,
        ),
    )
    inference_config = BaseConfig(inference_config_dict)
