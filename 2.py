# 처음 코드
import os
import csv
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time  # 시간 측정용 모듈 추가

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# CSV 파일 경로
csv_file = "labels.csv"

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

# 텍스트 임베딩 계산
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# 이미지와 텍스트의 유사도 계산
def get_image_features(image_path):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    
    return image_features

# 유사도가 높은 이미지 추천 함수
def recommend_similar_images(user_image_path):
    # 사용자 이미지의 특징 벡터 계산
    user_image_features = get_image_features(user_image_path)
    
    similarities = []
    for i, image_path in enumerate(image_paths):
        # 이미지 특징 벡터 계산
        image_features = get_image_features(image_path)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(user_image_features.cpu().numpy(), image_features.cpu().numpy())
        similarities.append((image_path, similarity[0][0]))
    
    # 유사도가 높은 이미지 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 5개의 유사한 이미지 반환
    return similarities[:5]

# 사용자 이미지 경로 입력
user_image_path = "TestImage/수국.jpg"

# 실행 시간 측정 시작
start_time = time.time()

# 추천된 이미지 출력
recommended_images = recommend_similar_images(user_image_path)

# 실행 시간 측정 종료
end_time = time.time()
elapsed_time = end_time - start_time

print("Recommended Images:")
for img, sim in recommended_images:
    print(f"Image: {img}, Similarity: {sim:.4f}")

print(f"\nTime taken for recommendation: {elapsed_time:.2f} seconds")
