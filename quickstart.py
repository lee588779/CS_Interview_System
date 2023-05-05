from google.cloud import speech

# 서비스 계정 키의 경로를 환경 변수로 설정 (공백으로 두면 안됨)
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "서비스계정키이름.json"

# 클라이언트 객체 생성
client = speech.SpeechClient()

# 음성 파일의 경로와 이름을 지정 (app.py에서 받은 파일 이름으로 수정)
audio_file = request.files

# 음성 파일을 바이너리 형식으로 읽기
with open(audio_file, "rb") as f:
    content = f.read()
    audio = speech.RecognitionAudio(content=content)

# 음성 인식 설정 (언어 코드, 타임스탬프 활성화 등)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code="ko-KR",
    enable_word_time_offsets=True,
)

# 음성 인식 요청 보내기
response = client.recognize(config=config, audio=audio)

# 음성 인식 결과 출력하기 (텍스트와 타임스탬프)
for result in response.results:
    alternative = result.alternatives[0]
    print("Transcript: {}".format(alternative.transcript))
    print("Confidence: {}".format(alternative.confidence))

    for word_info in alternative.words:
        word = word_info.word
        start_time = word_info.start_time
        end_time = word_info.end_time
        print(
            f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}"
        )