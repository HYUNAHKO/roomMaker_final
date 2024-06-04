import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# 예시 인수 확인
controlnet_model = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg")
print(controlnet_model.forward.__doc__)
