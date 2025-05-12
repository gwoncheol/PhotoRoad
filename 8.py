#이미지 + 텍스트 사용해서 유사도 계산 후 추천천
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
csv_file = "labels.csv"
embedding_cache_file = "image_embeddings.pkl"

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
    # 임베딩 저장
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(image_features, f)
    print("✅ 이미지 임베딩 저장 완료.")
else:
    print("✅ 저장된 이미지 임베딩 로딩 중...")
    with open(embedding_cache_file, "rb") as f:
        image_features = pickle.load(f)

# 유사도 기반 이미지 추천 (이미지 + 텍스트 임베딩 조합)
def recommend_similar_images_with_text(user_image_path, top_k=5, alpha=0.5):
    """
    alpha: 이미지 유사도 비중 (0.0 ~ 1.0). 나머지는 텍스트 유사도에 할당.
    """
    # 사용자 이미지 임베딩 계산
    image = preprocess(Image.open(user_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        user_image_feat = model.encode_image(image).cpu().numpy()
        user_image_feat /= np.linalg.norm(user_image_feat)

        # 텍스트 임베딩들과 비교하여 텍스트 유사도 계산
        user_text_sim = cosine_similarity(user_image_feat, text_features.cpu().numpy())[0]
    
    similarities = []
    for i, (path, img_feat) in enumerate(image_features.items()):
        img_feat_norm = img_feat / np.linalg.norm(img_feat)

        # 이미지 유사도
        img_sim = cosine_similarity(user_image_feat, img_feat_norm)[0][0]
        # 텍스트 유사도: 해당 이미지의 라벨 기준
        txt_sim = user_text_sim[i]

        # 이미지 유사도와 텍스트 유사도 조합
        total_sim = alpha * img_sim + (1 - alpha) * txt_sim
        similarities.append((path, total_sim, text_labels[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 실행
user_image_path = "TestImage/오프로드.jpg"  # 테스트 이미지 경로

start_time = time.time()
recommended_images = recommend_similar_images_with_text(user_image_path, top_k=5, alpha=0.5)
end_time = time.time()

print("\n📌 Recommended Images:")
for img, sim, txt in recommended_images:
    print(f"Image: {img}, Label: {txt}, Similarity: {sim:.4f}")

print(f"\n🕒 Recommendation step time: {end_time - start_time:.2f} seconds")

# 전체 실행 시간 종료
total_end_time = time.time()
print(f"✅ Total execution time: {total_end_time - total_start_time:.2f} seconds")
