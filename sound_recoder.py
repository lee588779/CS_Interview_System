import pyaudio
import wave

# 녹음 관련 상수 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5 # 녹음 시간 (초)
WAVE_FILE = "file.wav" # 녹음 파일 이름

# pyaudio 객체 생성
p = pyaudio.PyAudio()

# 스트림 열기
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 녹음 시작
print("Start to record the audio.")
frames = [] # 녹음된 데이터를 저장할 리스트

# 지정된 시간 동안 녹음
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# 녹음 종료
print("Recording is finished.")
stream.stop_stream()
stream.close()
p.terminate()

# 녹음된 데이터를 wav 파일로 저장
wf = wave.open(WAVE_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()