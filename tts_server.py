# tts_server.py をプロジェクトルートに作成
"""
TTSを担当する独立したプロセス
UDPでテキストを受け取って音声を再生する
"""
import socket
import sys
sys.path.insert(0, '.')
from voice.assistant import play_tts

TTS_PORT = 19000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", TTS_PORT))
print(f"[TTS Server] ポート {TTS_PORT} で待機中...")

while True:
    data, _ = sock.recvfrom(4096)
    text = data.decode("utf-8")
    print(f"[TTS Server] 再生: {text}")
    play_tts(text)