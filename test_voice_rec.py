from vosk import Model, KaldiRecognizer
import pyaudio
import json

model = Model("vosk-model-cn-0.22")
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)

print("开始录音（说中文）...")
while True:
    data = stream.read(2048)
    if rec.AcceptWaveform(data):
        print("识别结果:", json.loads(rec.Result())["text"])
    else:
        print("部分结果:", json.loads(rec.PartialResult())["partial"])
