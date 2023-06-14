# -*- coding:utf-8 -*-
import urllib3
import json
import base64

# openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Pronunciation"  # 영어
openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor" # 한국어

questions = [
    "디자인 패턴은 무엇인가요?",
    "싱글톤 패턴은 무엇인가요?",
]

answers = {}

answers["디자인 패턴은 무엇인가요?"] = "디자인 패턴은 프로그램을 설계할 때 발생했던 문제점들을 객체 간의 상호 관계 등을 이용하여 해결할 수 있도록 하나의 규약 형태로 만들어 놓은 것을 의미합니다."
answers["싱글톤 패턴은 무엇인가요?"] = "싱글톤 패턴은 클래스의 인스턴스가 오직 하나만 생성되도록 보장하고, 어디서든지 그 인스턴스에 접근할 수 있도록 하는 디자인 패턴입니다."

# wavanswer = answers[questions]

accessKey = "3f063dfd-cdea-415e-b7e9-0f92360351d3"
audioFilePath = "recorded.wav"
languageCode = "korean"
# script = "디자인 패턴은 프로그램을 설계할 때 발생했던 문제점들을 객체 간의 상호 관계 등을 이용하여 해결할 수 있도록 하나의 규약 형태로 만들어 놓은 것을 의미합니다."

file = open(audioFilePath, "rb")
audioContents = base64.b64encode(file.read()).decode("utf8")
file.close()

requestJson = {
    "argument": {
        "language_code": languageCode,
        # "script": wavanswer,
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

score = json.loads(str(response.data, "utf-8"))['return_object']['score']
#
# print("[responseCode] " + str(response.status))
# print("[responBody]")
# print(str(response.data, "utf-8"))
print(score)