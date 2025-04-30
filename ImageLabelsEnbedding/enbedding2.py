import os
import torch
import clip
from PIL import Image
import pickle
import csv

# 경로 설정
image_folder = "JejuImage"           # 이미지 폴더 경로
csv_file = "ImageLabelsEnbedding/labels2.csv"             # 텍스트 라벨이 포함된 CSV 파일
embedding_cache_file = "ImageLabelsEnbedding/image_embeddings3.pkl"  # 임베딩을 저장할 pkl 파일

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
print("⏳ CLIP 모델 로드 중...")
model, preprocess = clip.load("ViT-B/32", device)
print("✅ CLIP 모델 로드 완료.")

# 이미지 경로 및 텍스트 라벨 리스트 초기화
image_paths = []
text_labels = []

# CSV 파일 읽기
with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        image_path = os.path.join(image_folder, row[0])
        image_paths.append(image_path)
        text_labels.append(row[1])

print(f"🔍 {len(image_paths)}개의 이미지와 {len(text_labels)}개의 텍스트 라벨을 찾았습니다.")

# 이미지 임베딩 계산
print("⏳ 이미지 임베딩 계산 시작...")
image_features = {}
for path in image_paths:
    try:
        image = Image.open(path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input).cpu().numpy()
        image_features[path] = image_feature
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패: {path}, 에러: {e}")

# 텍스트 임베딩 계산
print("⏳ 텍스트 임베딩 계산 시작...")
text_inputs = clip.tokenize(text_labels).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs).cpu().numpy()
print(f"✅ 텍스트 임베딩 계산 완료: {len(text_features)}개의 텍스트 임베딩")

# 임베딩 및 라벨을 pkl 파일로 저장
try:
    with open(embedding_cache_file, "wb") as f:
        pickle.dump((image_features, text_features, text_labels), f)
    print(f"✅ 이미지 및 텍스트 임베딩 저장 완료: {embedding_cache_file}")
except Exception as e:
    print(f"⚠️ 임베딩 저장 실패, 에러: {e}")
