from flask import Flask, render_template, request
from google.cloud import speech
import sounddevice as sd
import soundfile as sf
import re

from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer, util
from konlpy.tag import Kkma
import numpy as np
from scipy.spatial.distance import jaccard, euclidean
import urllib3
import json
import base64

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

app = Flask(__name__)

# CS 기술 면접 문제 목록
questions = [
    "배열과 연결 리스트의 차이점은 무엇인가요?",
    "이진 탐색 트리 (Binary Search Tree)에 대해 설명하세요.",
    "해시 함수는 무엇인가요?",
    "스택의 특징은 무엇인가요?",
    "다이나믹 프로그래밍 (Dynamic Programming)이 무엇인가요?",
    "데이터베이스 정규화는 무엇인가요?",
    "JOIN연산은 무엇인가요?",
    "SQL 인덱스는 무엇인가요?",
    "트랜잭션은 무엇인가요?",
    "NoSQL 데이터베이스는 무엇인가요?",
    "객체 지향 프로그래밍은 무엇인가요?",
    "RESTful API는 무엇인가요?",
    "싱글톤 패턴은 무엇인가요?",
    "옵저버 패턴은 무엇인가요?",
    "팩토리 메서드 패턴은 무엇인가요?",
    "물리 계층은 무엇인가요?",
    "데이터 링크 계층은 무엇인가요?",
    "네트워크 계층은 무엇인가요?",
    "전송 계층은 무엇인가요?",
    "세션 계층은 무엇인가요?",
    "HTML은 무엇인가요?",
    "CSS는 무엇인가요?",
    "크로스 사이트 스크립팅 공격은 무엇인가요?",
    "프레임워크는 무엇인가요?",
    "라이브러리는 무엇인가요?",
    "서버리스 컴퓨팅은 무엇인가요?",
    "API 게이트웨이는 무엇인가요?",
    "그래프와 트리의 차이점은 무엇인가요?",
    "데이터베이스 샤딩은 무엇인가요?",
    "REST는 무엇인가요?",
    "마이크로서비스 아키텍처는 무엇인가요?",
    "지도 학습은 무엇인가요?",
    "비지도 학습은 무엇인가요?",
    "강화 학습은 무엇인가요?",
    "과적합은 무엇인가요?",
    "신경망은 무엇인가요?",
    "클라우드 컴퓨팅은 무엇인가요?",
    "가용성은 무엇인가요?",
    "도커는 무엇인가요?",
    "온디맨드는 무엇인가요?",
    "클라우드 스토리지는 무엇인가요?",
    "네이티브 앱은 무엇인가요?",
    "하이브리드 앱은 무엇인가요?",
    "Swift는 무엇인가요?",
    "Kotlin은 무엇인가요?",
    "웹 앱은 무엇인가요?",
    "취약점 스캐닝은 무엇인가요?",
    "펜 테스트는 무엇인가요?",
    "SIEM은 무엇인가요?",
    "멀티팩터 인증이 무엇인가요?",
    "웜은 무엇인가요?"
]

# 정답 문장과 사용자가 답변한 문장을 리스트로 저장할 딕셔너리
answers = {}

# 정답 문장을 딕셔너리에 저장
answers["배열과 연결 리스트의 차이점은 무엇인가요?"] = "배열은 고정된 크기를 가지고 연결 리스트는 동적인 크기를 가진다."
answers["이진 탐색 트리 (Binary Search Tree)에 대해 설명하세요."] = "각 노드가 최대 2개의 하위 노드를 가질 수 있는 구조입니다."
answers["해시 함수는 무엇인가요?"] = "해시 함수는 데이터를 입력으로 받아 고정 길이의 문자열로 변환하는 함수입니다."
answers["스택의 특징은 무엇인가요?"] = "스택은 마지막에 삽입된 데이터가 가장 먼저 제거되는 자료구조입니다"
answers["다이나믹 프로그래밍 (Dynamic Programming)이 무엇인가요?"] = "최적화 문제를 해결하는 컴퓨터 과학 및 알고리즘 설계 기법입니다."
answers["데이터베이스 정규화는 무엇인가요?"] = "데이터베이스 테이블을 구조화하고 중복 데이터를 최소화하여 데이터의 일관성과 무결성을 유지하는 기법입니다."
answers["JOIN연산은 무엇인가요?"] = "두 개 이상의 테이블에서 데이터를 결합하는 작업입니다."
answers["SQL 인덱스는 무엇인가요?"] = "관계형 데이터베이스에서 데이터 검색 및 조회 성능을 향상시키는데 사용되는 자료 구조입니다."
answers["트랜잭션은 무엇인가요?"] = "데이터베이스 작업의 단위입니다."
answers["NoSQL 데이터베이스는 무엇인가요?"] = "다양한 형태의 비관계형 데이터 모델을 기반으로 구축된 데이터베이스입니다."
answers["객체 지향 프로그래밍은 무엇인가요?"] = "객체 간의 상호작용을 통해 프로그램을 설계하고 구현하는 방법론입니다."
answers["RESTful API는 무엇인가요?"] = "웹 서비스를 설계하고 구현하는데 사용되는 아키텍처 스타일 및 원칙의 집합입니다."
answers["싱글톤 패턴은 무엇인가요?"] = "하나의 인스턴스만 생성되도록 보장하는 패턴입니다."
answers["옵저버 패턴은 무엇인가요?"] = "객체 사이의 일대다 의존 관계를 정의하여 한 객체의 상태 변경이 다른 객체들에게 통지되는 패턴입니다."
answers["팩토리 메서드 패턴은 무엇인가요?"] = "객체 생성을 서브 클래스에서 처리하도록 하는 패턴입니다."
answers["물리 계층은 무엇인가요?"] = "데이터를 전송하기 위한 물리적 매체와 하드웨어를 다루는 계층입니다."
answers["데이터 링크 계층은 무엇인가요?"] = "물리적 매체 상에서 오류 검출 및 수정, 흐름 제어, 노드 간의 직접 통신을 관리하는 계층입니다."
answers["네트워크 계층은 무엇인가요?"] = "데이터 패킷을 목적지로 라우팅하고 논리적 주소를 할당하는 계층입니다."
answers["전송 계층은 무엇인가요?"] = "송신자와 수신자 간의 신뢰성 있는 데이터 전송을 관리하는 계층입니다."
answers["세션 계층은 무엇인가요?"] = "데이터 교환을 관리하고 통신 세션을 설정, 유지 및 종료하는 계층입니다."
answers["HTML은 무엇인가요?"] = "웹 페이지를 생성하는데 사용되는 표준 마크업 언어입니다."
answers["CSS는 무엇인가요?"] = "HTML문서의 스타일, 레이아웃 및 디자인을 정의하는 스타일 시트 언어입니다."
answers["크로스 사이트 스크립팅 공격은 무엇인가요?"] = "악의적인 스크립트를 웹 페이지에 삽입하고 이 스크립트가 사용자의 브라우저에서 실행되게 하는 공격입니다."
answers["프레임워크는 무엇인가요?"] = "소프트웨어 개발을 위한 구조와 지침을 제공하는 개발 환경입니다."
answers["라이브러리는 무엇인가요?"] = "개발 중인 소프트웨어에 기능 또는 코드를 재사용할 수 있게 하는 코드의 집합입니다."
answers["서버리스 컴퓨팅은 무엇인가요?"] = "해시 테이블은 무한에 가까운 데이터들을 유한한 개수의 해시 값으로 매핑한 테이블입니다."
answers["API 게이트웨이는 무엇인가요?"] = "그래프는 정점과 간선으로 이루어진 자료 구조를 말하며 트리는 그래프 중 하나로 그래프의 특징처럼 정점과 간선으로 이루어져 있고 트리 구조로 배열된 일종의 계층적 데이터의 집합입니다."
answers["데이터베이스 샤딩은 무엇인가요?"] = "대규모 데이터베이스를 분할하고 여러 서버 또는 데이터 스토어에 분산 저장하는 기술입니다."
answers["REST는 무엇인가요?"] = "월드 와이드 웹에서 정보를 교환하고 상호작용하는데 사용되는 웹 서비스 디자인 원칙입니다."
answers["마이크로서비스 아키텍처는 무엇인가요?"] = "소프트웨어 애플리케이션을 작고 독립적인 기능 단위로 나누는 아키텍처 패턴입니다."
answers["지도 학습은 무엇인가요?"] = "레이블이 지정된 데이터에서 학습하여 입력과 출력 사이의 관계를 모델링하는 방법입니다."
answers["비지도 학습은 무엇인가요?"] = "레이블이 지정되지 않은 데이터에서 패턴, 구조 및 관계를 발견하고 모델링하는 방법입니다."
answers["강화 학습은 무엇인가요?"] = "에이전트가 환경과 상호작용하면서 어떤 작업을 수행하고 시간이 지남에 따라 보상을 최대화하기위한 최적의 행동 정책을 학습하는 방법입니다."
answers["과적합은 무엇인가요?"] = "모델이 훈련 데이터에 너무 맞춰져서 새로운 데이터에 대한 일반화 능력이 떨어지는 문제입니다."
answers["신경망은 무엇인가요?"] = "복잡한 데이터 패턴을 학습하고 인식하는데 사용되는 기계 학습 모델 중 하나입니다."
answers["클라우드 컴퓨팅은 무엇인가요?"] = "인터넷을 통해 IT 리소스와 서비스를 제공하고 사용하는 컴퓨팅 기술과 모델입니다."
answers["가용성은 무엇인가요?"] = "시스템, 서비스 또는 리소스가 사용 가능하거나 작동 중인 상태를 나타내는 개념입니다."
answers["도커는 무엇인가요?"] = "컨테이너화 기술을 기반으로하는 오픈 소스 플랫폼입니다."
answers["온디맨드는 무엇인가요?"] = "사용자가 필요한 것을 원하는 순간에 즉시 이용할 수 있는 방식을 말합니다."
answers["클라우드 스토리지는 무엇인가요?"] = "데이터를 인터넷을 통해 원격 서버에 저장하는 기술과 서비스를 말합니다."
answers["네이티브 앱은 무엇인가요?"] = "특정 플랫폼 또는 운영체제에 최적화된 방식으로 개발된 모바일 앱을 말합니다."
answers["하이브리드 앱은 무엇인가요?"] = "네이티브 앱과 웹 앱의 특징을 혼합한 앱으로 다양한 플랫폼에서 동작하도록 설계합니다."
answers["Swift는 무엇인가요?"] = "애플이 개발한 프로그래밍 언어로 iOS 및 Linux운영체제에서 개발을 위해 사용됩니다."
answers["Kotlin은 무엇인가요?"] = "자바 가상 머신과 안드로이드 플랫폼을 대상으로 하는 프로그래밍 언어입니다."
answers["웹 앱은 무엇인가요?"] = "웹 브라우저를 통해 접근되고 실행되는 소프트웨어 응용 프로그램입니다."
answers["취약점 스캐닝은 무엇인가요?"] = "컴퓨터 시스템, 네트워크, 소프트웨어에서 보안 취약점을 탐지하고 식별하는 프로세스입니다."
answers["펜 테스트는 무엇인가요?"] = "기술적 환경의 보안 취약점을 확인하고 해결하기 위해 명시적으로 인증된 해커가 시스템에 침투하고 공격을 모방하는 보안 테스트 프로세스입니다."
answers["SIEM은 무엇인가요?"] = "조직의 IT 인프라에서 발생하는 보안 관련 이벤트와 데이터를 수집, 분석, 모니터링하고 보고하는 역할을 수행하는 보안 도구입니다."
answers["멀티팩터 인증이 무엇인가요?"] = "사용자가 시스템 또는 온라인 서비스에 로그인할 때 여러 인증 요소를 사용하여 보안을 강화하는 보안 메커니즘입니다."
answers["웜은 무엇인가요?"] = "다른 컴퓨터 및 네트워크 시스템으로 자동으로 복제 및 전파될 수 있는 자체 복제형 맬웨어입니다."


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
    answer = ""
    # 결과에서 첫 번째 대안의 텍스트를 가져옴
    for result in response.results:
        answer = result.alternatives[0].transcript
        break

    # 전처리 전의 문장들을 리스트로 저장
    sentences = [question, answer]

    # 문장들을 벡터화함
    corpus_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    # 코사인 유사도 계산
    cosine_sim = util.pytorch_cos_sim(corpus_embeddings[0], corpus_embeddings[1]).item()

    kkma = Kkma()

    tok1 = kkma.morphs(question)
    tok2 = kkma.morphs(answer)
    set1 = set(tok1)
    set2 = set(tok2)
    intersec = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersec
    jaccard_sim = intersec/union

    # jaccard_sim =jaccard(corpus_embeddings[0], corpus_embeddings[1])

    uc_dis = euclidean(corpus_embeddings[0], corpus_embeddings[1])

    cosjcd_mix = (cosine_sim*0.5)+(jaccard_sim*0.5)

    accessKey = "key"
    languageCode = "Korean"

    with open(filename, "rb") as f:
        audioContents = base64.b64encode(f.read()).decode("utf8")

    requestJson = {
        "argument": {
            "language_code": languageCode,
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
    return render_template('answer.html', question=question, answers=answers, answer=answer, cosine_sim=cosine_sim, jaccard_sim=jaccard_sim, uc_dis=uc_dis, cosjcd_mix=cosjcd_mix, correct_answer=answers[question],
                           score=score)

@app.route('/')
def dictionary():
    return render_template('dictionary.html')

if __name__ == '__main__':
    app.run()
