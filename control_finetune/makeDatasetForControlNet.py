from datasets import load_from_disk
import json
import os
from datasets import Dataset, load_dataset
from PIL import Image
import torch

def load_image(file_path):
    return Image.open(file_path).convert('RGB')

def load_data(base_path, captions_file):
    images = []
    captions = []
    with open(captions_file, 'r') as file:
        captions_dict = json.load(file)

    # 이미지 파일을 순회하며 데이터셋 생성
    for img_file in os.listdir(base_path):
        if img_file.endswith('clean.png'):
            clean_path = os.path.join(base_path, img_file)
            depth_path = os.path.join(base_path, img_file.replace('clean', 'depth'))
            segmentation_path = os.path.join(base_path, img_file.replace('clean', 'segmentation'))
            
            # 이미지 로드
            clean_image = load_image(clean_path)
            depth_image = load_image(depth_path)
            segmentation_image = load_image(segmentation_path)

            # 이미지 이름으로 캡션 찾기
            if img_file in captions_dict:
                caption = captions_dict[img_file]

                # 이미지와 캡션 추가
                images.append({'clean': clean_image, 'depth': depth_image, 'segmentation': segmentation_image})
                captions.append(caption)

    return Dataset.from_dict({'image': images, 'caption': captions})

# 데이터셋 경로와 캡션 파일 경로 설정
base_path = '/home/work/roomMaker/clean_images'
captions_file = '/home/work/roomMaker/caption/captions.json'

# 데이터 로드
dataset = load_data(base_path, captions_file)

# 데이터셋을 디스크에 저장
dataset.save_to_disk('controlNet_model/controlNetDataset')

# 로컬에서 데이터셋 로드
dataset = load_from_disk('/home/work/roomMaker/controlNet_model/controlNetDataset')

# Push the dataset to the Hugging Face Hub
dataset.push_to_hub("controlnetdataset")

