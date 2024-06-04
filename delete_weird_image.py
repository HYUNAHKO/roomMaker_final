import os
import json

# 경로 설정
base_dir = "/home/work/roomMaker"
dataset_dir = os.path.join(base_dir, "dataset/airbnb")
generated_image_dir = os.path.join(base_dir, "hyuna_controlNet_model/generated_image")
clean_image_dir = os.path.join(base_dir, "images/clean_images")
depth_image_dir = os.path.join(base_dir, "images/depth_images")
segmentation_image_dir = os.path.join(base_dir, "images/segmentation_images")
caption_file_path = os.path.join(base_dir, "caption/caption.json")

# dataset/airbnb 폴더의 파일 이름 수집 (확장자 제거)
dataset_files = set(f.replace(".jpg", "") for f in os.listdir(dataset_dir) if f.endswith(".jpg"))

# 캡션 파일 로드
with open(caption_file_path, 'r') as f:
    captions = json.load(f)

# 삭제 대상 파일 확인 및 삭제 함수
def delete_unmatched_files(target_dir, suffix):
    for filename in os.listdir(target_dir):
        if filename.endswith(suffix):
            base_filename = filename.replace(suffix, "")
            if base_filename not in dataset_files:
                file_path = os.path.join(target_dir, filename)
                print(f"Deleting file: {file_path}")
                os.remove(file_path)

# generated_image 폴더에서 삭제
delete_unmatched_files(generated_image_dir, "generated.png")

# clean_images 폴더에서 삭제
delete_unmatched_files(clean_image_dir, "clean.png")

# depth_images 폴더에서 삭제
delete_unmatched_files(depth_image_dir, "depth.png")

# segmentation_images 폴더에서 삭제
delete_unmatched_files(segmentation_image_dir, "segmentation.png")

# captions.json 파일에서 삭제
if isinstance(captions, list):
    keys_to_delete = [caption for caption in captions if caption["file_name"].replace(".jpg", "") not in dataset_files]
    for caption in keys_to_delete:
        print(f"Deleting caption for file: {caption['file_name']}")
        captions.remove(caption)
else:
    print("Error: Expected captions to be a list.")

# 수정된 캡션 파일 저장
with open(caption_file_path, 'w') as f:
    json.dump(captions, f, indent=4)
