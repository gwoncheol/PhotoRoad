from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json

# BLIP 모델 및 프로세서 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 이미지 → 캡션 함수
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 폴더 내 이미지 전체 처리
def caption_images_in_folder(folder_path, save_path="captions.json"):
    captions = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, file)
            try:
                caption = generate_caption(full_path)
                captions[file] = caption
                print(f"[✔] {file}: {caption}")
            except Exception as e:
                print(f"[✘] {file} 실패: {e}")

    # 결과 JSON 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 캡션 결과가 {save_path}에 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    image_folder = "./images"  # 이미지 폴더 경로
    caption_images_in_folder(image_folder)
