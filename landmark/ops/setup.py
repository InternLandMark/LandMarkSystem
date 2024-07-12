import os
from distutils.sysconfig import get_python_inc

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

kernel_dir = "./kernel/"
python_env_include_path = get_python_inc()
cutlass_dir = os.environ.get("CUTLASS_DIR", "../../3rdparty/cutlass/")
print("python_env_include_path:", python_env_include_path)
print("cutlass_dir:", cutlass_dir)


setup(
    name="grid_sample_ndhwc2ncdhw_3d",
    ext_modules=[
        CUDAExtension(
            "grid_sample_ndhwc2ncdhw_3d",
            sources=[
                kernel_dir + "densityappfeature/grid_sampler_3d_ndhwc.cpp",
                kernel_dir + "densityappfeature/grid_sampler_3d_ndhwc_kernel.cu",
            ],
            include_dirs=[
                python_env_include_path,
                os.path.join(cutlass_dir, "util/include"),
                os.path.join(cutlass_dir, "include"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-std=c++17",
                    "--gpu-architecture=sm_80",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
