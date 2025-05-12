from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os

# 이미지 경로 설정
IMAGE_DIR = "./images"  # 이미지가 있는 폴더 경로
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 프로세서 로드 (blip2-opt-2.7b)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.to(DEVICE)

# 결과 저장
output_file = "blip2_captions.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(IMAGE_DIR, filename)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(DEVICE, torch.float16 if DEVICE == "cuda" else torch.float32)
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            print(f"{filename}: {generated_text}")
            f.write(f"{filename}: {generated_text}\n")

print("✅ BLIP-2 캡션 생성 완료! 결과는 'blip2_captions.txt'에 저장되었습니다.")
