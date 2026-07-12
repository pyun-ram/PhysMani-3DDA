from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_cuda',
    ext_modules=[
        CUDAExtension(
            'fps_cuda',
            ['src/fps.cpp', 'src/fps_kernel.cu', 'src/fps_kernel_with_mask.cu',],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)