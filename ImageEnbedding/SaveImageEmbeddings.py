import os
import torch
import clip
from PIL import Image
import pickle

# JejuImage 폴더 경로
image_folder = "JejuImage"

# 저장할 디렉터리 절대 경로
output_dir = "C:/Users/dlxlr/Desktop/PhotoRoad2/ImageEnbedding"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

# 저장할 pkl 파일 전체 경로
embedding_cache_file = os.path.join(output_dir, "image_embeddings.pkl")

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device)

# 이미지 경로 리스트 만들기
image_paths = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image_paths.append(os.path.join(image_folder, filename))

print(f"{len(image_paths)}개의 이미지를 찾았습니다.")

# 이미지 임베딩 계산
image_features = {}

print("이미지 임베딩 계산 시작")
for path in image_paths:
    try:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image).cpu().numpy()
        image_features[path] = feat
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패: {path}, 에러: {e}")

# pkl 파일로 저장
with open(embedding_cache_file, "wb") as f:
    pickle.dump(image_features, f)

print(f"✅ 이미지 임베딩 저장 완료: {embedding_cache_file}")
