import os
import json
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 모델 및 프로세서 로드
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = pipeline.to(device)
clip_model = clip_model.to(device)

# 이미지, 마스크, 캡션 파일 디렉토리 설정
image_dir = "/home/work/roomMaker/images/clean_images"
mask_dir = "/home/work/roomMaker/images/segmentation_images"
caption_file = "/home/work/roomMaker/caption/captions.json"

# 캡션 파일 로드
with open(caption_file, 'r') as f:
    captions = json.load(f)

# 캡션 확인 (디버깅용)
for key, value in captions.items():
    print(f"Filename: {key}, Caption: {value}")

# 이미지 텐서를 [0, 1] 범위로 정규화하는 함수
def normalize_image_tensor(image_tensor):
    return image_tensor * 0.5 + 0.5

# 이미지 디렉토리의 파일을 순회
for filename in os.listdir(image_dir):
    if filename.endswith("clean.png"):  # .png 파일만 처리
        base_filename = filename.replace("clean.png", "")  # .png 확장자 제거
        image_path = os.path.join(image_dir, filename)
        mask_filename = base_filename + "segmentation.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(image_path) and os.path.exists(mask_path):
            clean_image = Image.open(image_path).convert("RGB")
            segmentation_mask = Image.open(mask_path).convert("L")

            # 캡션 파일에서 해당 이미지의 캡션을 가져옴
            caption_key = base_filename + ".jpg"  # 캡션 파일은 .jpg 확장자를 사용
            caption = captions.get(caption_key, "Room with furniture")
            print(f"Processing file: {filename}, Caption: {caption}")

            inputs = processor(images=clean_image, return_tensors="pt", padding=True).to(device)
            text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
            text_inputs = clip_model.text_model(**text_inputs)

            # 이미지 텐서를 정규화하고 GPU로 이동
            normalized_image_tensor = normalize_image_tensor(inputs["pixel_values"]).to(device)
            segmentation_mask = torch.tensor(np.array(segmentation_mask)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

            # 모델 입력 조합
            model_input = {
                "prompt": [caption],  # 프롬프트는 문자열 또는 문자열 리스트
                "image": normalized_image_tensor,
                "mask_image": segmentation_mask
            }

            # 이미지 생성
            generated_images = pipeline(**model_input)
            generated_image = generated_images.images[0]
            save_path = os.path.join("control_finetune/new_generation", base_filename + "generated.png")
            generated_image.save(save_path)
        else:
            print(f"Skipping {filename} as the image or mask file does not exist.")
