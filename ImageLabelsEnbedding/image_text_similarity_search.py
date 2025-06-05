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
model, preprocess = clip.load("ViT-B/32", device=device)

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

def get_image_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(inp).cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb

def get_text_embedding(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens).cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb

def recommend_similar_images(user_image_path=None, user_text=None, top_k=5, alpha_img=0.5, alpha_text=0.5):
    # 이미지 임베딩
    if user_image_path:
        image_emb = get_image_embedding(user_image_path)
    else:
        image_emb = None

    # 텍스트 임베딩
    if user_text and user_text.strip() != "":
        text_emb = get_text_embedding(user_text)
    else:
        text_emb = None

    sims = []
    for i, (path, ifeat) in enumerate(image_features.items()):
        ifeat_norm = ifeat / np.linalg.norm(ifeat)

        sim_img = 0
        sim_text = 0

        if image_emb is not None:
            sim_img = cosine_similarity(image_emb, ifeat_norm)[0][0]

        if text_emb is not None:
            sim_text = cosine_similarity(text_emb, ifeat_norm)[0][0]

        # 임베딩이 둘 다 있으면 가중치 조합, 아니면 존재하는 것만 사용
        if (image_emb is not None) and (text_emb is not None):
            total_sim = alpha_img * sim_img + alpha_text * sim_text
        elif image_emb is not None:
            total_sim = sim_img
        elif text_emb is not None:
            total_sim = sim_text
        else:
            raise ValueError("이미지와 텍스트 입력 둘 다 비어있습니다.")

        sims.append((path, total_sim, text_labels[i]))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

def main():
    print("=== CLIP 이미지+텍스트 통합 추천 시스템 ===")
    user_image_path = input("이미지 경로 입력 (없으면 엔터): ").strip()
    if user_image_path == "":
        user_image_path = None
    else:
        if not os.path.exists(user_image_path):
            print("❌ 이미지 경로가 존재하지 않습니다.")
            return

    user_text = input("텍스트 입력 (없으면 엔터): ").strip()
    if user_text == "":
        user_text = None

    if user_image_path is None and user_text is None:
        print("❌ 이미지 또는 텍스트 중 최소 하나는 입력해야 합니다.")
        return

    top_k = input("추천 개수 (기본 5): ").strip()
    top_k = int(top_k) if top_k.isdigit() else 5

    alpha_img = 0.5
    alpha_text = 0.5

    if user_image_path and user_text:
        print("이미지와 텍스트 모두 입력됨, 유사도 가중치 조정")
        alpha_img_input = input("이미지 가중치 alpha_img (0~1, 기본 0.5): ").strip()
        alpha_text_input = input("텍스트 가중치 alpha_text (0~1, 기본 0.5): ").strip()
        try:
            alpha_img = float(alpha_img_input) if alpha_img_input != "" else 0.5
            alpha_text = float(alpha_text_input) if alpha_text_input != "" else 0.5
            s = alpha_img + alpha_text
            if s != 0:
                alpha_img /= s
                alpha_text /= s
        except:
            print("가중치 입력이 잘못되어 기본값 사용")

    start = time.time()
    results = recommend_similar_images(user_image_path, user_text, top_k=top_k, alpha_img=alpha_img, alpha_text=alpha_text)
    end = time.time()

    print("\n✅ 추천 결과:")
    for path, sim, label in results:
        print(f"이미지: {path}, 라벨: {label}, 유사도: {sim:.4f}")
    print(f"\n⏱️ 추천 소요 시간: {end - start:.2f}초")

if __name__ == "__main__":
    main()
