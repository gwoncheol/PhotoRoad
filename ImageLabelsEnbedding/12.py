import os
import torch
import clip
from PIL import Image
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import time

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device)

# pkl 및 csv 파일 경로
embedding_cache_file = "ImageLabelsEnbedding/image_embeddings3.pkl"
csv_file = "ImageLabelsEnbedding/labels2.csv"

# 임베딩 로드
with open(embedding_cache_file, "rb") as f:
    data = pickle.load(f)

if isinstance(data, tuple) and len(data) == 3:
    image_features, text_features, text_labels = data
elif isinstance(data, tuple) and len(data) == 2:
    image_features, text_features = data
    text_labels = []
    with open(csv_file, newline="", encoding="utf-8") as f_csv:
        reader = csv.reader(f_csv)
        next(reader)
        for row in reader:
            text_labels.append(row[1])
else:
    raise ValueError("예상치 못한 pkl 구조")

# 추천 함수
def recommend_similar_images_with_text(user_image_path, top_k=5, alpha=0.5):
    img = Image.open(user_image_path).convert("RGB")
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        ufeat = model.encode_image(inp).cpu().numpy()
    ufeat /= np.linalg.norm(ufeat)

    txt_sims = cosine_similarity(ufeat, text_features)[0]

    sims = []
    for i, (path, ifeat) in enumerate(image_features.items()):
        ifeat /= np.linalg.norm(ifeat)
        img_sim = cosine_similarity(ufeat, ifeat)[0][0]
        txt_sim = txt_sims[i]
        total_sim = alpha * img_sim + (1 - alpha) * txt_sim
        sims.append((path, total_sim, text_labels[i]))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# 추천 실행
user_image_path = "TestImage/카페.jpg"
start = time.time()
results = recommend_similar_images_with_text(user_image_path, top_k=3, alpha=0.7)
end = time.time()

# 결과 출력
print("\n추천 결과:")
for path, sim, label in results:
    print(f"이미지: {path}, 라벨: {label}, 유사도: {sim:.4f}")
print(f"추천 소요 시간: {end - start:.2f}초")
