## 🛠️ 필수 패키지 설치

이 프로젝트를 실행하기 위해서는 다음과 같은 Python 패키지들이 필요합니다.

### 1. **PyTorch**
CLIP 모델과 텐서 연산을 위해 사용됩니다. GPU를 사용하는 경우 CUDA 버전에 맞게 설치해야 합니다.

```bash
pip install torch torchvision torchaudio
````

> 💡 **CUDA 환경에 맞는 PyTorch 설치는 아래 링크를 참고하세요:**
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

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

```

---
