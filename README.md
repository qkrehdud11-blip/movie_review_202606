# 🎬 AI Movie Recommendation System

## 📌 프로젝트 소개

**2017년 ~ 2022년 영화 리뷰 데이터**를 기반으로 사용자가 선택한 영화 또는 입력한 키워드와 유사한 영화를 추천하는 **AI 영화 추천 시스템**입니다.

리뷰 데이터를 전처리한 뒤 `TF-IDF`, `Cosine Similarity`, `Word2Vec`를 활용하여 영화 제목 기반 추천과 키워드 기반 추천을 제공합니다.
또한 `PyQt5`를 이용해 GUI 프로그램으로 구현하여 사용자가 쉽게 영화 추천 결과를 확인할 수 있도록 제작했습니다.

---

## 🛠 개발 환경

| 항목            | 내용                        |
| ------------- | ------------------------- |
| Language      | Python 3.12               |
| GUI           | PyQt5                     |
| NLP           | KoNLPy Okt                |
| Vectorization | TF-IDF                    |
| Embedding     | Word2Vec                  |
| Similarity    | Cosine Similarity         |
| Visualization | WordCloud, Matplotlib     |
| Data          | Movie Reviews 2017 ~ 2022 |

---

## 📁 프로젝트 구조

```bash
movie_recommendation/
│
├── datasets/
│   ├── reviews_2017_2022.csv
│   ├── reviews_2017_2022_test.csv
│   └── stopwords.csv
│
├── models/
│   ├── tfidf.pkl
│   ├── Tfidf_movie_review.mtx
│   └── word2vec_movie_review.model
│
├── movie_recommendation.ui
├── movie_recommendation_app.py
│
├── job01_preprocessing.py
├── job02_word_cloud.py
├── job03_TFIDF.py
├── job04_recommendation.py
├── job05_word2vec.py
│
├── requirements.txt
└── README.md
```

---

## 📊 데이터 구성

`datasets` 폴더를 생성한 뒤, 영화 리뷰 데이터를 넣어야 합니다.

```bash
mkdir datasets
mkdir models
```

리뷰 데이터 파일:

```bash
datasets/reviews_2017_2022.csv
```

데이터 컬럼은 반드시 아래와 같이 구성합니다.

| 컬럼명     | 설명    |
| ------- | ----- |
| titles  | 영화 제목 |
| reviews | 영화 리뷰 |

예시:

```csv
titles,reviews
겨울왕국,영상미가 좋고 노래가 인상적인 애니메이션 영화
기생충,스토리 전개가 강렬하고 사회적 메시지가 뛰어난 영화
```

---

## 📦 requirements.txt

아래 내용을 `requirements.txt`로 저장합니다.

```txt
contourpy==1.3.3
cycler==0.12.1
fonttools==4.63.0
gensim==4.4.0
joblib==1.5.3
jpype1==1.7.1
kiwisolver==1.5.0
konlpy==0.6.0
lxml==6.1.1
matplotlib==3.10.9
narwhals==2.22.1
numpy==2.4.6
packaging==26.2
pandas==3.0.3
pillow==12.2.0
pyparsing==3.3.2
PyQt5==5.15.11
PyQt5-Qt5==5.15.19
PyQt5_sip==12.18.0
python-dateutil==2.9.0.post0
scikit-learn==1.9.0
scipy==1.17.1
six==1.17.0
smart_open==7.6.1
threadpoolctl==3.6.0
wordcloud==1.9.6
wrapt==2.2.1
```

라이브러리 설치:

```bash
pip install -r requirements.txt
```

---

## 🔎 주요 기능

### 1. 리뷰 텍스트 전처리

영화 리뷰에서 한글만 추출하고, Okt 형태소 분석기를 이용해 주요 품사만 남깁니다.

추출 품사:

* 명사
* 동사
* 형용사

```python
review = re.sub('[^가-힣]', ' ', review)
tokened_review = okt.pos(review, stem=True)
```

불용어와 한 글자 단어를 제거하여 추천에 필요한 핵심 단어만 사용합니다.

---

### 2. TF-IDF 기반 영화 제목 추천

선택한 영화의 리뷰 벡터와 전체 영화 리뷰 벡터를 비교하여 유사도가 높은 영화를 추천합니다.

```python
cosine_sim = linear_kernel(Tfidf_matrix[movieIdx], Tfidf_matrix)
```

---

### 3. Word2Vec 기반 키워드 추천

사용자가 키워드를 입력하면 Word2Vec 모델에서 유사 단어를 찾고, 해당 단어들을 조합하여 추천 문장을 생성합니다.

```python
sim_word = embedding_model.wv.most_similar(keyword, topn=10)
```

생성된 문장을 TF-IDF 벡터로 변환한 뒤, 전체 영화 리뷰와 비교하여 관련성이 높은 영화를 추천합니다.

---

### 4. WordCloud 시각화

특정 영화 리뷰에 자주 등장하는 단어를 WordCloud로 시각화하여 영화별 리뷰 특징을 확인할 수 있습니다.

```python
wordcloud = WordCloud(
    font_path=font_path,
    background_color='white'
).generate_from_frequencies(worddict)
```

---

### 5. PyQt5 GUI 프로그램

사용자는 GUI에서 영화 제목을 선택하거나 키워드를 입력하여 추천 결과를 받을 수 있습니다.

GUI 구성 요소:

| 구성 요소    | 기능       |
| -------- | -------- |
| ComboBox | 영화 제목 선택 |
| LineEdit | 키워드 입력   |
| Button   | 추천 실행    |
| Label    | 추천 결과 출력 |

---

## ⚙ 실행 방법

### 1. 가상환경 생성

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

### 2. 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

### 3. 폴더 생성

```bash
mkdir datasets
mkdir models
```

---

### 4. 데이터 준비

아래 파일을 `datasets` 폴더에 넣습니다.

```bash
datasets/reviews_2017_2022.csv
datasets/stopwords.csv
```

---

### 5. 전처리 실행

```bash
python preprocessing.py
```

생성 파일:

```bash
datasets/reviews_2017_2022_test.csv
```

---

### 6. TF-IDF 모델 생성

```bash
python tfidf_model.py
```

생성 파일:

```bash
models/tfidf.pkl
models/Tfidf_movie_review.mtx
```

---

### 7. Word2Vec 모델 생성

```bash
python word2vec_model.py
```

생성 파일:

```bash
models/word2vec_movie_review.model
```

---

### 8. GUI 실행

```bash
python movie_recommendation_app.py
```

---

## 🧠 추천 시스템 동작 흐름

```text
리뷰 데이터 로드
        ↓
한글 정제
        ↓
Okt 형태소 분석
        ↓
명사 / 동사 / 형용사 추출
        ↓
불용어 제거
        ↓
TF-IDF 벡터 생성
        ↓
Word2Vec 모델 학습
        ↓
영화 제목 또는 키워드 입력
        ↓
코사인 유사도 계산
        ↓
추천 영화 출력
```

---

## ✅ 추천 방식

### 영화 제목 기반 추천

```text
영화 제목 선택
        ↓
해당 영화 리뷰 벡터 추출
        ↓
전체 영화 리뷰 벡터와 유사도 비교
        ↓
유사도가 높은 영화 추천
```

---

### 키워드 기반 추천

```text
키워드 입력
        ↓
Word2Vec 유사 단어 추출
        ↓
키워드 + 유사 단어 기반 추천 문장 생성
        ↓
TF-IDF 벡터 변환
        ↓
전체 영화 리뷰와 유사도 비교
        ↓
관련 영화 추천
```

---

## 📌 핵심 코드

### 영화 제목 기반 추천

```python
def recommendation_by_title(self, title):
    movieIdx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
    cosine_sim = linear_kernel(self.Tfidf_matrix[movieIdx], self.Tfidf_matrix)

    recommendations = self.getRecommendation(cosine_sim)
    recommendations = '\n'.join(recommendations[1:])

    return recommendations
```

---

### 키워드 기반 추천

```python
def recommendation_by_keyword(self, keyword):
    try:
        sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
    except:
        return '제가 모르는 단어에요 ㅠㅠ'

    sentence = [keyword] * 11
    count = 10

    for word, _ in sim_word:
        sentence = sentence + [word] * count
        count -= 1

    sentence = ' '.join(sentence)
    sentence_vec = self.Tfidf.transform([sentence])
    cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)

    recommendations = self.getRecommendation(cosine_sim)
    recommendations = '\n'.join(recommendations[:10])

    return recommendations
```

---

## 🖥 GUI 화면 특징

* Dark Mode 기반 디자인
* 보라색 포인트 컬러 적용
* 영화 제목 선택 ComboBox
* 키워드 입력창
* 추천 버튼
* 추천 결과 표시 영역
* 영화 제목 자동완성 기능 적용

---

## 🚨 코드 작성 시 주의사항

한글 정제 정규식은 아래처럼 작성해야 합니다.

```python
review = re.sub('[^가-힣]', ' ', review)
```

아래 코드는 권장하지 않습니다.

```python
review = re.sub('[^가~힣]', ' ', review)
```

또한 전처리된 데이터를 실제 추천 모델에 사용하려면 모델 학습 코드에서 아래 파일을 읽는 것이 좋습니다.

```python
df_reviews = pd.read_csv('./datasets/reviews_2017_2022_test.csv')
```

---

## 📈 프로젝트 결과

본 프로젝트를 통해 다음 기능을 구현했습니다.

* 영화 리뷰 데이터 전처리
* Okt 기반 한국어 형태소 분석
* 불용어 제거
* TF-IDF 기반 문서 벡터화
* 코사인 유사도 기반 영화 추천
* Word2Vec 기반 키워드 확장
* WordCloud 리뷰 시각화
* PyQt5 기반 GUI 영화 추천 프로그램 제작

---

## 💡 프로젝트 의의

이 프로젝트는 단순히 영화 제목이나 장르만 비교하는 방식이 아니라,
실제 리뷰 텍스트에 포함된 단어와 의미를 기반으로 유사 영화를 추천하도록 구현했습니다.

이를 통해 사용자는 특정 영화와 비슷한 분위기의 영화를 찾거나,
`감동`, `액션`, `겨울`, `로맨스` 같은 키워드를 입력해 관련성이 높은 영화를 추천받을 수 있습니다.

---

## 🔧 개선 가능 사항

* 영화 포스터 이미지 출력
* 추천 결과 클릭 시 영화 상세 정보 표시
* 감성 분석 기능 추가
* 중복 영화 제거 로직 강화
* 키워드 자동완성 기능 개선
* 사용자 평점 기반 추천 기능 추가
* BERT 기반 문장 임베딩 추천 방식 적용
* 추천 결과 저장 기능 추가

---

## 🏷 Tech Stack

```text
Python 3.12
Pandas
KoNLPy
Okt
Scikit-learn
TF-IDF
Cosine Similarity
Gensim
Word2Vec
WordCloud
Matplotlib
PyQt5
```

---

## 📌 프로젝트 한 줄 요약

**2017년부터 2022년까지의 영화 리뷰 데이터를 활용하여 TF-IDF와 Word2Vec 기반으로 유사 영화를 추천하는 PyQt5 GUI 영화 추천 시스템입니다.**
