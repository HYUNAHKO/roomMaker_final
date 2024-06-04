import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import ControlNetModel, AutoPipelineForInpainting, DEISMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 폴더 지정
dataset_path = "/home/work/roomMaker/dataset/airbnb"
segmentation_path = "/home/work/roomMaker/images/segmentation_images"
depth_path = "/home/work/roomMaker/images/depth_images"
caption_file = "/home/work/roomMaker/caption/caption.json"
output_path = "/home/work/roomMaker/output"
os.makedirs(output_path, exist_ok=True)

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_folder, segmentation_folder, depth_folder, caption_file, transform=None, depth_transform=None):
        self.image_folder = image_folder
        self.segmentation_folder = segmentation_folder
        self.depth_folder = depth_folder
        self.transform = transform
        self.depth_transform = depth_transform
        with open(caption_file, 'r') as f:
            self.captions = {item["file_name"]: item["caption"] for item in json.load(f)}
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        self.image_files = [f for f in self.image_files if self._valid_file(f)]
        if len(self.image_files) == 0:
            print("No valid data found!")

    def _valid_file(self, filename):
        base_filename = filename.replace(".jpg", "")
        segmentation_file = os.path.join(self.segmentation_folder, f"{base_filename}segmentation.png")
        depth_file = os.path.join(self.depth_folder, f"{base_filename}depth.png")
        if not os.path.exists(segmentation_file):
            print(f"Segmentation file missing for {filename}")
            return False
        if not os.path.exists(depth_file):
            print(f"Depth file missing for {filename}")
            return False
        if filename not in self.captions:
            print(f"Caption missing for {filename}")
            return False
        return True

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        base_filename = self.image_files[idx].replace(".jpg", "")
        segmentation_path = os.path.join(self.segmentation_folder, f"{base_filename}segmentation.png")
        depth_path = os.path.join(self.depth_folder, f"{base_filename}depth.png")
        image = Image.open(image_path).convert("RGB")
        segmentation = Image.open(segmentation_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        caption = self.captions[self.image_files[idx]]

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)
            depth = self.depth_transform(depth)

        return image, segmentation, depth, caption

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

depth_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 데이터셋 및 데이터로더 준비
dataset = CustomDataset(dataset_path, segmentation_path, depth_path, caption_file, transform, depth_transform)

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Please check the paths and files.")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ControlNet 모델 로드
controlnet_model = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16).to(device)

# Inpainting 모델 로드
inpainting_model = AutoPipelineForInpainting.from_pretrained('lykon/absolute-reality-1.6525-inpainting', torch_dtype=torch.float16, variant="fp16")
inpainting_model.scheduler = DEISMultistepScheduler.from_config(inpainting_model.scheduler.config)
inpainting_model = inpainting_model.to(device)

# 캡션 처리 모델 로드
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# 손실 함수 및 옵티마이저 설정
optimizer = torch.optim.Adam(controlnet_model.parameters(), lr=5e-5)
scheduler = DEISMultistepScheduler.from_config(controlnet_model.config)

# Fine-tuning 진행
num_epochs = 5
for epoch in range(num_epochs):
    controlnet_model.train()
    for batch in dataloader:
        images, segmentations, depths, captions = batch
        images = images.to(device).to(torch.float16)
        segmentations = segmentations.to(device).to(torch.float16)
        depths = depths.to(device).to(torch.float16)

        # 캡션 처리
        inputs = clip_processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        text_features = clip_model.get_text_features(**inputs)

        # ControlNet을 통한 세그멘테이션 예측
        controlnet_cond = torch.cat([segmentations, depths], dim=1)
        controlnet_outputs = controlnet_model(sample=images, timestep=torch.tensor(1, dtype=torch.float16, device=device), encoder_hidden_states=text_features, controlnet_cond=controlnet_cond)

        # 손실 계산 및 역전파
        loss = controlnet_outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")

# Fine-tuning 완료된 모델 저장
controlnet_model.save_pretrained("/home/work/roomMaker/fine_tuned_controlnet")

# Fine-tuned 모델을 사용한 Inpainting 예제
controlnet_model.eval()
test_image = dataset[0][0].unsqueeze(0).to(device).to(torch.float16)
test_segmentation = dataset[0][1].unsqueeze(0).to(device).to(torch.float16)
test_depth = dataset[0][2].unsqueeze(0).to(device).to(torch.float16)
test_caption = dataset[0][3]

# 세그멘테이션 및 인페인팅 수행
conditioning = torch.cat([test_segmentation, test_depth], dim=1)  # Combine segmentation and depth as conditioning
controlnet_outputs = controlnet_model(sample=test_image, timestep=torch.tensor(1, dtype=torch.float16, device=device), encoder_hidden_states=text_features, controlnet_cond=conditioning)
inpainted_image = inpainting_model(prompt=test_caption, image=test_image, mask_image=controlnet_outputs['logits'])[0]

# 결과 저장
result_image = transforms.ToPILImage()(inpainted_image.cpu().squeeze())
result_image.save(os.path.join(output_path, "fine_tuned_result.jpg"))
print("Fine-tuning completed and result image saved.")
