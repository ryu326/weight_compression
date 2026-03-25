from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="my_kernels_cuda",
    ext_modules=[
        CUDAExtension(
            name="my_kernels",
            sources=[
                "src/wrapper.cpp",
                "src/my_torch_ans.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                    "--ptxas-options=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
