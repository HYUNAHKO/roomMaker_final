import os
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

# 데이터셋 폴더 지정
dataset_path = "/home/work/roomMaker/dataset/airbnb"

# 이미지 파일 로딩 함수
def load_images_from_folder(folder):
    images = []
    file_names = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            with open(path, 'rb') as f:
                images.append(Image.open(f).convert('RGB'))
                file_names.append(filename)  # 파일 이름 저장
    return {"image": images, "file_name": file_names}

# 데이터 로딩
rooms = load_images_from_folder(dataset_path)

# 이미지 전처리
def pad_image(image, target_size):
    """이미지의 비율을 유지하면서 패딩을 추가하여 target_size로 변환"""
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image.resize(target_size)

transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_image(img, (256, 256))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 이미지 데이터에 캡션 추가하는 함수
def caption_image_data(images, file_names):
    captions = []
    for image, file_name in zip(images, file_names):
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        captions.append({"file_name": file_name, "caption": caption})
    return captions

# 모든 이미지에 대해 캡션 처리
captions = caption_image_data(rooms["image"], rooms["file_name"])

# 결과 JSON 파일로 저장
output_json_path = "/home/work/roomMaker/caption/caption.json"

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(captions, f, ensure_ascii=False, indent=4)

print("All images processed and captions saved to JSON file.")
