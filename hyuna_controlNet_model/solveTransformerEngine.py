import ctypes
import os

# Attempt to load the library directly
library_path = "/usr/local/lib/python3.10/dist-packages/libtransformer_engine.so"
ctypes.CDLL(library_path)

# Import your required module after the library is loaded
from diffusers import DiffusionPipeline
