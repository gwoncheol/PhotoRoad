#전체 실행 시간도 출력함함
import os
import csv
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle  # 임베딩 저장용

# 전체 실행 시간 시작
total_start_time = time.time()

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# CSV 파일 경로
csv_file = "ImageEnbedding/labels.csv"
embedding_cache_file = "ImageEnbedding/image_embeddings.pkl"

# 이미지 경로 및 텍스트 설명을 저장할 리스트
image_paths = []
text_labels = []

# CSV 파일 읽기
with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        image_paths.append(row[0])
        text_labels.append(row[1])

# 텍스트 라벨을 CLIP 모델에 입력 가능한 형식으로 변환
text_inputs = clip.tokenize(text_labels).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# 이미지 임베딩 미리 계산 및 저장
if not os.path.exists(embedding_cache_file):
    print("⏳ 이미지 임베딩 미리 계산 중...")
    image_features = {}
    for path in image_paths:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image).cpu().numpy()
        image_features[path] = feat
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(image_features, f)
    print("✅ 이미지 임베딩 저장 완료.")
else:
    print("✅ 저장된 이미지 임베딩 로딩 중...")
    with open(embedding_cache_file, "rb") as f:
        image_features = pickle.load(f)

# 유사도 기반 이미지 추천
def recommend_similar_images(user_image_path, top_k=5):
    image = preprocess(Image.open(user_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        user_feat = model.encode_image(image).cpu().numpy()

    similarities = []
    for path, feat in image_features.items():
        sim = cosine_similarity(user_feat, feat)[0][0]
        similarities.append((path, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 실행
user_image_path = "TestImage/car.jpg"

start_time = time.time()
recommended_images = recommend_similar_images(user_image_path)
end_time = time.time()

print("\n추천 결과:")
for img, sim in recommended_images:
    # 이미지 경로에 대응되는 텍스트 설명 찾기
    try:
        index = image_paths.index(img)
        label = text_labels[index]
    except ValueError:
        label = "설명 없음"
    print(f"Image: {img}, Text: \"{label}\", Similarity: {sim:.4f}")

# 전체 실행 시간 종료
total_end_time = time.time()
print(f"추천 소요 시간: {total_end_time - total_start_time:.2f} seconds")

# 문제의 원인은 이미지 임베딩을 저장할 때 사용한 image_features의 경로가 절대 경로로 저장되고, 
# labels.csv에서 읽은 image_paths는 상대 경로로 저장되기 때문에 경로 불일치가 발생하는 것입니다. 
# 이로 인해 추천된 이미지 경로를 image_paths.index(img)로 찾을 때 ValueError가 발생하여 "설명 없음"이 출력됩니다. 
# 해결 방법은 두 가지로, 첫째는 이미지 임베딩 시 경로를 상대 경로로 저장하고, 
# 둘째는 labels.csv에서 읽어온 경로를 절대 경로로 변환하는 것입니다.
