from landmark.nerf_components.model_components.fields.encodings import (
    BranchParallelTensorVMEncoding,
    ChannelParallelTensorVMEncoding,
)
from landmark.nerf_components.model_components.fields.encodings.hash_encoding import (
    BranchParallelHashEncoding,
)
from landmark.nerf_components.model_components.fields.fused_anchor_decoder import (
    FusedAnchorDecoder,
)
from landmark.nerf_components.model_components.fields.fused_mlp_decoder import (
    FusedMLPDecoder,
)
from landmark.nerf_components.ray_samplers.fused_uniform_sampler import (
    FusedUniformSampler,
)

# Mapping of encoding types to their corresponding channel parallel converters.
channel_parallel_convert_map = {
    "TensorVMEncoding": ChannelParallelTensorVMEncoding,
}

# Mapping of encoding types to their corresponding branch parallel converters.
branch_parallel_convert_map = {
    "TensorVMEncoding": BranchParallelTensorVMEncoding,
    "HashEncoding": BranchParallelHashEncoding,
}

# Mapping of encoding types to their corresponding element parallel converters.
elem_parallel_convert_map = {}

# Main mapping of parallelization strategies to their respective converter maps.
encoding_convert_map = {
    "ChannelParallel": channel_parallel_convert_map,
    "BranchParallel": branch_parallel_convert_map,
    "ElemParallel": elem_parallel_convert_map,
}

# Mapping of kernel fusion components to their fused counterparts.
kernel_fusion_components_convert_map = {
    "MLPDecoder": FusedMLPDecoder,
    # "VolumeRenderer": FusedVolumeRenderer,
    # "TensorVMEncoding": FusedTensorVMEncoding,
    # "ChannelParallelTensorVMEncoding": FusedChannelParallelTensorVMEncoding,
    "UniformSampler": FusedUniformSampler,
    "AnchorDecoder": FusedAnchorDecoder,
}
