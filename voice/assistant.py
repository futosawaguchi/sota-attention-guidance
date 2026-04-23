import os, wave, io, time, threading, queue, re
import numpy as np
import sounddevice as sd
import requests
from openai import OpenAI
from pydub import AudioSegment
import webrtcvad
import config

# ===== Azure設定 =====
API_KEY    = config.AZURE_OPENAI_API_KEY
AZURE_BASE = config.AZURE_BASE_URL
API_VER    = "2025-03-01-preview"

if not API_KEY or not AZURE_BASE:
    raise ValueError(".envファイルにAZURE_API_KEYとAZURE_BASE_URLを設定してください")

STT_DEPLOY  = config.AZURE_STT_DEPLOY
CHAT_DEPLOY = config.AZURE_CHAT_DEPLOY
TTS_DEPLOY  = config.AZURE_TTS_DEPLOY

STT_URL = f"{AZURE_BASE}/openai/deployments/{STT_DEPLOY}/audio/transcriptions?api-version={API_VER}"
TTS_URL = f"{AZURE_BASE}/openai/deployments/{TTS_DEPLOY}/audio/speech?api-version={API_VER}"

client = OpenAI(base_url=f"{AZURE_BASE}/openai/v1", api_key=API_KEY)

# ===== マイク／VAD設定 =====
SR            = 16000
FRAME_MS      = 30
FRAME_SAMPLES = SR * FRAME_MS // 1000
VAD_MODE      = 2
vad           = webrtcvad.Vad(VAD_MODE)

# ===== 動的変更可能なパラメータ =====
silence_sec    = 1.2
extra_wait_sec = 1.5
max_retry      = 2

# ===== 状態 =====
is_ai_speaking    = threading.Event()
barge_in_event    = threading.Event()
waiting_for_extra = threading.Event()
raw_queue         = queue.Queue()
extra_audio_queue = queue.Queue()
processing_lock   = threading.Lock()

chat_history = [
    {"role": "system", "content": "あなたは日本語で簡潔に答えるアシスタントです。"}
]
history_lock = threading.Lock()


# ---------------------------------------------------------------
# 完結判定（ローカルルールベース・ゼロレイテンシ）
# ---------------------------------------------------------------
_INCOMPLETE_PATTERNS = [
    # フィラー・接続詞「だけ」で終わる（文頭ではなく全体がそれだけの場合）
    r"^(しかし|でも|そして|それで|それから|あの|えっと|えー|んー|うーん|まあ|あー|ところで|さて)[、。,.\s]*$",
    # 読点終わり
    r"[、,]\s*$",
    # 助詞終わり（「は」「へ」を除外 → 「こんにちは」「こんにちへ」誤爆防止）
    r"[をがもにとでのから]\s*$",
    r"[をがもにとでのから][、。]\s*$",
    # 接続助詞終わり
    r"(けど|けれど|けれども|ので|から、|ながら|つつ|つつも)[、,]?\s*$",
    # 「〜について」など話題提示
    r"(について|に関して|に対して|としては)[、,]?\s*$",
]

def is_complete(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    for pattern in _INCOMPLETE_PATTERNS:
        if re.search(pattern, t):
            print(f"  [Judge] 「{t}」→ false（{pattern}）")
            return False
    print(f"  [Judge] 「{t}」→ true")
    return True


# ---------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------
def frames_to_wav_bytes(frames: list) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    resp = requests.post(
        STT_URL,
        headers={"api-key": API_KEY},
        files={"file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"model": "gpt-4o-transcribe", "language": "ja",
              "response_format": "json", "temperature": 0},
        timeout=30
    )
    if resp.status_code != 200:
        print(f"[STT Error] {resp.status_code}")
        return ""
    return (resp.json().get("text") or "").strip()


def has_speech_in_frames(frames: list) -> bool:
    if not frames:
        return False
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"  [Speech Check] RMS={rms:.1f}")
    return rms > 80


# ---------------------------------------------------------------
# 追加録音
# ---------------------------------------------------------------
def drain_extra_audio(extra_sec: float) -> list:
    while not extra_audio_queue.empty():
        try:
            extra_audio_queue.get_nowait()
        except queue.Empty:
            break
    waiting_for_extra.set()
    extra_frames = []
    deadline = time.time() + extra_sec
    while time.time() < deadline:
        try:
            frame = extra_audio_queue.get(timeout=deadline - time.time())
            extra_frames.append(frame)
        except queue.Empty:
            break
    waiting_for_extra.clear()
    return extra_frames


# ---------------------------------------------------------------
# TTS再生
# ---------------------------------------------------------------
def play_tts(text: str):
    resp = requests.post(
        TTS_URL,
        headers={"api-key": API_KEY, "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini-tts", "input": text, "voice": "nova"},
        timeout=30
    )
    if resp.status_code != 200:
        print(f"[TTS Error] {resp.status_code}")
        return

    seg = AudioSegment.from_mp3(io.BytesIO(resp.content))
    seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    pcm = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    is_ai_speaking.set()
    try:
        sd.play(pcm, samplerate=24000)
        sd.wait()  # バージイン検知なしで最後まで再生
    finally:
        is_ai_speaking.clear()


# ---------------------------------------------------------------
# 発話処理
# ---------------------------------------------------------------
def process_speech(initial_frames: list):
    if not processing_lock.acquire(blocking=False):
        return
    try:
        all_frames = initial_frames.copy()

        for attempt in range(max_retry + 1):
            wav  = frames_to_wav_bytes(all_frames)
            text = transcribe(wav)

            if not text:
                return

            print(f"\nあなた（{'暫定' if attempt == 0 else '再判定'}）: {text}")

            if is_complete(text):
                print(f"あなた（確定）: {text}")
                break

            if attempt < max_retry:
                print(f"  → 続きを待機中…（+{extra_wait_sec}秒）")
                extra = drain_extra_audio(extra_wait_sec)

                if not has_speech_in_frames(extra):
                    print("  → 追加は無音のみ。完結とみなします。")
                    break

                all_frames.extend(extra)
                print(f"  → {len(extra)}フレーム追加（声あり）")
            else:
                print("  → 最大リトライ到達。完結とみなします。")

        with history_lock:
            chat_history.append({"role": "user", "content": text})
            messages = chat_history.copy()

        resp = client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=messages
        )
        ai_text = resp.choices[0].message.content
        print(f"AI: {ai_text}\n")

        with history_lock:
            chat_history.append({"role": "assistant", "content": ai_text})

        play_tts(ai_text)

    finally:
        processing_lock.release()


# ---------------------------------------------------------------
# VADメインループ
# ---------------------------------------------------------------
def is_echo_frame(frame_bytes: bytes) -> bool:
    if not is_ai_speaking.is_set():
        return False
    # Sota発話中は全フレームを無視
    return True  # ← 閾値判定をやめて常にTrueに


def audio_callback(indata, frames, time_info, status):
    data = bytes(indata)
    raw_queue.put(data)
    if waiting_for_extra.is_set():
        extra_audio_queue.put(data)


def vad_loop():
    speech_frames = []
    silent_count  = 0
    in_speech     = False

    def silence_frames_threshold():
        return int(silence_sec * 1000 / FRAME_MS)

    with sd.RawInputStream(
        samplerate=SR, blocksize=FRAME_SAMPLES,
        dtype="int16", channels=1, callback=audio_callback
    ):
        print("会話を開始します（Ctrl+C で終了）")
        print(f"無音閾値: {silence_sec}秒 / 追加待機: {extra_wait_sec}秒 / 最大リトライ: {max_retry}回\n")

        while True:
            frame = raw_queue.get()
            if is_echo_frame(frame):
                continue

            try:
                is_speech = vad.is_speech(frame, SR)
            except Exception:
                continue

            if is_speech:
                if not in_speech:
                    in_speech = True
                    if is_ai_speaking.is_set():
                        barge_in_event.set()
                speech_frames.append(frame)
                silent_count = 0
            else:
                if in_speech:
                    silent_count += 1
                    speech_frames.append(frame)
                    if silent_count >= silence_frames_threshold():
                        if len(speech_frames) >= 8:
                            t = threading.Thread(
                                target=process_speech,
                                args=(speech_frames.copy(),),
                                daemon=True
                            )
                            t.start()
                        speech_frames = []
                        silent_count  = 0
                        in_speech     = False


# ---------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        vad_loop()
    except KeyboardInterrupt:
        print("\n終了しました")