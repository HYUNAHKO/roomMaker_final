import os
import json
import torch
from PIL import Image
from torchvision import transforms
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

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

# LLaVA 모델 경로 설정
llava_model_path = "/home/work/roomMaker/llava-llama-3-8b-v1_1-int4.gguf"
mmproj_model_path = "/home/work/roomMaker/llava-llama-3-8b-v1_1-mmproj-f16.gguf"

# LLaVA 모델 로드
handler = Llava15ChatHandler(clip_model_path=mmproj_model_path, verbose=True)
model = Llama(
    model_path=llava_model_path,
    chat_handler=handler,
    n_ctx=2048,
    logits_all=True,
    n_gpu_layers=50  # GPU를 사용할 레이어 수를 지정
)

# 이미지 데이터에 캡션 추가하는 함수
def caption_image_data(images, file_names):
    captions = []
    for image, file_name in zip(images, file_names):
        # 이미지 전처리
        processed_image = transform(image)  # 배치 차원 제거

        # 텐서를 PIL 이미지로 변환
        processed_image_pil = transforms.ToPILImage()(processed_image)

        # 이미지를 byte array로 변환
        from io import BytesIO
        image_bytes = BytesIO()
        processed_image_pil.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()

        # LLaVA 모델을 사용하여 캡션 생성
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_bytes", "image_bytes": {"bytes": image_bytes}},
                {"type" : "text", "text": "Describe this image"}
            ]}
        ]
        result = model.create_chat_completion(messages=messages)

        # 결과에서 캡션 추출
        caption = result['choices'][0]['message']['content']
        captions.append({"file_name": file_name, "caption": caption})
        
    return captions

# 모든 이미지에 대해 캡션 처리
captions = caption_image_data(rooms["image"], rooms["file_name"])

# 결과 JSON 파일로 저장
output_json_path = "/home/work/roomMaker/caption/caption2.json"

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(captions, f, ensure_ascii=False, indent=4)

print("All images processed and captions saved to JSON file.")
