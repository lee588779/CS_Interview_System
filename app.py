from flask import Flask, render_template, request
from google.cloud import speech
import sounddevice as sd
import soundfile as sf
import re

from konlpy.tag import Okt
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import urllib3
import json
import base64

app = Flask(__name__)

# CS 기술 면접 문제 목록
questions = [
    "디자인 패턴은 무엇인가요?",
    "싱글톤 패턴은 무엇인가요?",
    "What is inheritance?",
    "What is encapsulation?",
    "What is abstraction?"
]

# 정답 문장과 사용자가 답변한 문장을 리스트로 저장할 딕셔너리
answers = {}

# 정답 문장을 딕셔너리에 저장
answers["디자인 패턴은 무엇인가요?"] = "디자인 패턴은 프로그램을 설계할 때 발생했던 문제점들을 객체 간의 상호 관계 등을 이용하여 해결할 수 있도록 하나의 규약 형태로 만들어 놓은 것을 의미합니다."
answers["싱글톤 패턴은 무엇인가요?"] = "싱글톤 패턴은 클래스의 인스턴스가 오직 하나만 생성되도록 보장하고, 어디서든지 그 인스턴스에 접근할 수 있도록 하는 디자인 패턴입니다."

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"

# 메인 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html', questions=questions)

# 문제 선택 페이지 라우팅
@app.route('/select')
def select():
    # 선택한 문제의 인덱스를 받아옴
    index = request.args.get('index')
    # 인덱스에 해당하는 문제를 가져옴
    question = questions[int(index)]
    return render_template('select.html', question=question)

@app.route('/answer', methods=['GET'])
def answer():
    # GET 요청에서 question 인자를 받아옴
    question = request.args.get('question')
    # 마이크로부터 음성 데이터를 캡처함
    fs = 16000  # 샘플링 주파수
    duration = 20  # 녹음 시간 (초)
    print("Recording...")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # 녹음이 끝날 때까지 대기
    print("Done")
    # 녹음한 음성 데이터를 wav파일로 저장함
    filename = "recorded.wav"
    sf.write(filename, data, fs)
    # 구글 stt api 클라이언트 생성함
    client = speech.SpeechClient()
    # wav파일을 열어서 바이너리 데이터로 읽어옴
    with open(filename, "rb") as f:
        content = f.read()
    # 음성 인식 요청을 생성함
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fs,
        language_code="ko-KR",
    )
    # 음성 인식 요청을 보내고 결과를 받아옴
    response = client.recognize(config=config, audio=audio)
    # 결과에서 첫 번째 대안의 텍스트를 가져옴
    for result in response.results:
        answer = result.alternatives[0].transcript
        break

    # 전처리 전의 문장들을 리스트로 저장
    sentences = [question, answer]

    # 전처리 후의 문장들을 저장할 빈 리스트 생성
    preprocessed_sentences = []

    # 불용어 리스트 생성
    f = open('korean.txt', 'r', encoding='utf-8')
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]
    f.close()

    # 어간 추출기 생성
    stemmer = PorterStemmer()

    # 표제어 추출기 생성
    lemmatizer = WordNetLemmatizer()

    # 형태소 분석기 생성 (KoNLPy의 Okt 사용)
    tokenizer = Okt()

    # 문장들에 대해 반복문 실행
    for sentence in sentences:
        # 대소문자 통일: 모두 소문자로 변환
        sentence = sentence.lower()

        # 구두점 제거: 정규식을 이용하여 구두점 제거
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # 숫자 제거: 정규식을 이용하여 숫자 제거
        sentence = re.sub(r'\d+', '', sentence)

        # 불용어 제거: 불용어 리스트를 이용하여 불용어 제거
        sentence = ' '.join([word for word in sentence.split() if word not in stop_words])

        # 어간 추출: 어간 추출기를 이용하여 단어의 어근 찾기
        sentence = ' '.join([stemmer.stem(word) for word in sentence.split()])

        # 표제어 추출: 표제어 추출기를 이용하여 단어의 기본형 찾기
        sentence = ' '.join([lemmatizer.lemmatize(word) for word in sentence.split()])

        # 형태소 분석: 형태소 분석기를 이용하여 단어와 품사를 추출
        sentence = tokenizer.pos(sentence)

        # 명사만 추출: 품사가 명사인 단어만 선택
        sentence = [word for word, tag in sentence if tag == 'Noun']

        # 전처리 후의 문장 리스트에 추가
        preprocessed_sentences.append(sentence)

    # 문장들을 단어 단위로 벡터화
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(preprocessed_sentences)

    # 문장들을 n-gram으로 벡터화 (n=2)
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform([' '.join(sentence) for sentence in preprocessed_sentences])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(X)

    # 자카드 유사도 계산
    jaccard_sim = jaccard_score(X[0].toarray()[0], X[1].toarray()[0], average='micro')

    # 코사인 유사도와 자카드 유사도의 평균 계산
    average_sim = (cosine_sim[0][1] + jaccard_sim) / 2

    accessKey = "key"
    languageCode = "korean"
    # script = answers[question]

    with open(filename, "rb") as f:
        audioContents = base64.b64encode(f.read()).decode("utf8")

    requestJson = {
        "argument": {
            "language_code": languageCode,
            # "script": script,
            "audio": audioContents
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8", "Authorization": accessKey},
        body=json.dumps(requestJson)
    )

    # 응답 본문을 JSON 객체로 파싱
    score = json.loads(str(response.data, "utf-8"))['return_object']['score']

    # answer.html 파일을 렌더링하여 응답함
    return render_template('answer.html', question=question, answers=answers, answer=answer, cosine_sim=cosine_sim, jaccard_sim=jaccard_sim, average_sim=average_sim, correct_answer=answers[question], score=score)

if __name__ == '__main__':
    app.run()
