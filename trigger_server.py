import os
import io
import wave
import time
import queue
import socket
import threading
import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv

load_dotenv()

# ===== Azure STT設定 =====
API_KEY    = os.getenv("AZURE_API_KEY")
AZURE_BASE = os.getenv("AZURE_BASE_URL")
API_VER    = "2025-03-01-preview"
STT_DEPLOY = os.getenv("AZURE_STT_DEPLOY")
STT_URL    = f"{AZURE_BASE}/openai/deployments/{STT_DEPLOY}/audio/transcriptions?api-version={API_VER}"

# ===== トリガーワード =====
TRIGGER_PHRASES = [
    "見てください",
    "これを見て",
    "見て",
    "あれを見て",
    "そこを見て",
]

# ===== UDP設定（main.pyへ通知） =====
TRIGGER_HOST = "127.0.0.1"
TRIGGER_PORT = 19001  # tts_serverは19000なので別ポート

# ===== VAD設定 =====
SR            = 16000
FRAME_MS      = 30
FRAME_SAMPLES = SR * FRAME_MS // 1000
VAD_MODE      = 2
silence_sec   = 1.0

vad       = webrtcvad.Vad(VAD_MODE)
raw_queue = queue.Queue()
sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def frames_to_wav_bytes(frames: list) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    try:
        resp = requests.post(
            STT_URL,
            headers={"api-key": API_KEY},
            files={"file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"model": "gpt-4o-transcribe", "language": "ja",
                  "response_format": "json", "temperature": 0},
            timeout=30
        )
        if resp.status_code != 200:
            return ""
        return (resp.json().get("text") or "").strip()
    except Exception as e:
        print(f"[Trigger] STTエラー: {e}")
        return ""


def is_trigger(text: str) -> bool:
    """トリガーワードが含まれているか判定"""
    return any(phrase in text for phrase in TRIGGER_PHRASES)


def send_trigger():
    """main.pyにトリガーをUDP送信"""
    sock.sendto(b"TRIGGER", (TRIGGER_HOST, TRIGGER_PORT))
    print("[Trigger] トリガー送信!")


def audio_callback(indata, frames, time_info, status):
    raw_queue.put(bytes(indata))


def vad_loop():
    speech_frames = []
    silent_count  = 0
    in_speech     = False
    silence_threshold = int(silence_sec * 1000 / FRAME_MS)

    with sd.RawInputStream(
        samplerate=SR, blocksize=FRAME_SAMPLES,
        dtype="int16", channels=1, callback=audio_callback
    ):
        print("[Trigger] トリガー検出待機中...")

        while True:
            frame = raw_queue.get()

            try:
                is_speech = vad.is_speech(frame, SR)
            except Exception:
                continue

            if is_speech:
                if not in_speech:
                    in_speech = True
                speech_frames.append(frame)
                silent_count = 0
            else:
                if in_speech:
                    silent_count += 1
                    speech_frames.append(frame)
                    if silent_count >= silence_threshold:
                        if len(speech_frames) >= 8:
                            wav   = frames_to_wav_bytes(speech_frames)
                            text  = transcribe(wav)
                            if text:
                                print(f"[Trigger] 認識: {text}")
                                if is_trigger(text):
                                    send_trigger()
                        speech_frames = []
                        silent_count  = 0
                        in_speech     = False


if __name__ == "__main__":
    try:
        vad_loop()
    except KeyboardInterrupt:
        print("\n[Trigger] 終了しました")