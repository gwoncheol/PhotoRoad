#벡터 값을 시각화 해서 볼 수 있는 코드드
import pickle
import numpy as np

with open("ImageEnbedding/image_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# 전체 출력
for img_name, vector in embeddings.items():
    print(f"\n📍 {img_name}")
    print(vector)  # vector는 보통 shape (1, 512)
