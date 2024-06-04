from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from datasets import Dataset
from PIL import Image as PILImage
import torch
import os
import json

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
                images.append(PILImage.open(f).convert('RGB'))
                file_names.append(filename)  # 파일 이름 저장
    # 리스트를 딕셔너리 형태로 변환
    return Dataset.from_dict({"image": images, "file_name": file_names})

# 데이터 로딩
rooms = load_images_from_folder(dataset_path)

# 모델 및 프로세서 로드
model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
processor = ViTFeatureExtractor.from_pretrained("ydshieh/vit-gpt2-coco-en")
tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 이미지 데이터에 캡션 추가하는 함수
def caption_image_data(batch):
    # 각 이미지를 순회하며 캡션을 생성
    captions = []
    for image in batch['image']:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)

    # 생성된 캡션을 배치 딕셔너리에 추가
    batch['image_caption'] = captions
    return batch

# 모든 이미지에 대해 캡션 처리
rooms_proc = rooms.map(caption_image_data, batched=True, batch_size=8)

# 결과 JSON 파일로 저장
output_json_path = "/home/work/roomMaker/caption/caption.json"

# 데이터셋에서 각 이미지 파일명과 해당 캡션을 추출하여 직접 JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as f:
    # 이미지 파일 이름과 캡션을 함께 저장
    json_data = {example['file_name']: example['image_caption'][0] for example in rooms_proc}
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print("All images processed and captions saved to JSON file.")
