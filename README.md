## 🛠️ 필수 패키지 설치

이 프로젝트를 실행하기 위해서는 다음과 같은 Python 패키지들이 필요합니다.

### 1. **PyTorch**

CLIP 모델과 텐서 연산을 위해 사용됩니다. GPU를 사용하는 경우 CUDA 버전에 맞게 설치해야 합니다.

```bash
pip install torch torchvision torchaudio
```

> 💡 **CUDA 환경에 맞는 PyTorch 설치는 아래 링크를 참고하세요:** > [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

### 2. **OpenAI CLIP**

OpenAI의 CLIP 모델을 로드하기 위한 패키지입니다.

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

### 3. **Pillow**

이미지 로딩 및 전처리를 위한 라이브러리입니다.

```bash
pip install Pillow
```

---

### 4. **NumPy**

이미지 및 텍스트 임베딩 벡터 연산에 필요합니다.

```bash
pip install numpy
```

---

### 5. **scikit-learn**

코사인 유사도 계산을 위해 사용됩니다.

```bash
pip install scikit-learn
```

---

### 6. **(선택 사항) tqdm**

진행률 표시 등에 유용한 라이브러리입니다. (임베딩 생성 시 사용 가능)

```bash
pip install tqdm
```

---

### ✅ 전체 패키지 한 번에 설치 (추천)

아래 명령어 하나로 모든 필수 패키지를 설치할 수 있습니다:

```bash
pip install torch torchvision torchaudio git+https://github.com/openai/CLIP.git Pillow numpy scikit-learn
```

`tqdm`은 필요 시 별도로 설치하세요.

````

---
물론입니다! 아래는 **GitHub용 `README.md` 마크다운 형식**으로 바로 복사해서 사용하실 수 있는 버전입니다:

```markdown
# 📌 PhotoRoad - 이미지 & 라벨 기반 유사도 계산 프로젝트

이 프로젝트는 **CLIP** 모델을 활용하여 이미지와 텍스트 라벨의 임베딩을 생성하고, 유사도를 기반으로 유사한 이미지를 찾는 시스템입니다.

---

## 📁 프로젝트 구조

```

PhotoRoad/
│
├── ImageEmbedding/ # ❌ (현재 사용 안 함)
│ └── # 이미지 임베딩만 사용하는 기능 (비활성화)
│
├── ImageLabelsEnbedding/ # ✅ 이미지 + 라벨 임베딩 기반 유사도 분석
│ ├── 12.py # 메인 실행 코드 (pkl 로딩 및 유사도 분석)
│ ├── embedding2.py # 이미지 & 라벨 임베딩 계산 후 pkl 저장
│ ├── image_text_similarity_search.py # 이미지, 텍스트 입력 받고 출력함
│ ├── labels2.csv # 이미지에 대한 라벨 데이터 (텍스트)
│ └── image_embeddings3.pkl # 저장된 임베딩 캐시 파일
│
├── TestImage/ # 테스트용 이미지 디렉토리
│
└── blip2_caption.py # 이미지에서 라벨 추출 (BLIP-2 기반)

```

---

## ⚙️ 주요 파일 설명

### `ImageLabelsEnbedding/12.py`
- `image_embeddings3.pkl`을 불러와 이미지와 라벨 간 유사도를 계산합니다.
- 테스트 이미지에 대해 가장 유사한 이미지나 라벨을 출력합니다.

### `ImageLabelsEnbedding/enbedding2.py`
- `labels2.csv`의 라벨과 이미지 경로를 이용해 CLIP 임베딩을 생성합니다.
- 생성된 임베딩은 `.pkl`로 저장되어 추후 유사도 계산에 사용됩니다.

### `labels2.csv`
- 각 이미지에 대한 텍스트 라벨 정보가 저장된 CSV 파일입니다.
- 형식 예: `image1.jpg,제주 해변`

### `blip2_caption.py`
- 이미지에서 자동으로 캡션(라벨)을 생성하는 코드입니다.
- **BLIP-2** 모델 기반으로 구현되었습니다.

---

## 🧪 테스트 이미지

- `TestImage/` 폴더에는 테스트용 이미지가 저장됩니다.
- 이 이미지를 사용하여 `12.py`에서 유사도 분석을 수행할 수 있습니다.

---

## 🚧 참고사항

- `ImageEmbedding/` 폴더는 이미지 임베딩만을 다루던 초기 버전이며 현재는 사용하지 않습니다.
- `.pkl` 파일이 50MB 이상인 경우 GitHub 업로드에 제한이 있을 수 있습니다.
  - 이 경우 [Git Large File Storage (LFS)](https://git-lfs.github.com) 사용을 권장합니다.

---

## 🧠 사용 모델

- **CLIP (ViT-B/32)**: 이미지 및 텍스트 임베딩 생성
- **BLIP-2 (선택적)**: 이미지에서 자동 캡션(라벨) 추출

---

## 🚀 실행 방법

```bash
# 1. 이미지 및 라벨 임베딩 생성
$ python ImageLabelsEnbedding/enbedding2.py

# 2. 유사도 계산 실행
$ python ImageLabelsEnbedding/12.py
````

---

## 💡 향후 확장 아이디어

- 이미지 유사도만 사용하는 기능 재활성화
- 사용자 정의 라벨 기반 추천 기능 추가
- Web 기반 이미지 검색 시스템 개발

---

## 📄 라이선스

이 프로젝트는 **비상업적 연구 및 교육 목적**에 한해 자유롭게 사용하실 수 있습니다.

```

---

이제 이 내용을 `README.md` 파일에 붙여넣고 커밋하면 바로 GitHub에서 잘 보입니다.
필요하면 커밋 메시지도 함께 도와드릴게요!
```
