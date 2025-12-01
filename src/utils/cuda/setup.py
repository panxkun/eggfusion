from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_tracking",
    version="0.1.0",
    author="Xiaokun Pan",
    author_email="panxkun@gmail.com",
    ext_modules=[
        CUDAExtension(
            name="cuda_tracking_ext",  # Changed name to avoid conflict
            sources=["src/tracking.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)