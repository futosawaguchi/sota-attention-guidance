import cv2
import time
import threading
import numpy as np
import config
from sota import controller

# ========== 状態管理 ==========
_tracking_enabled = True
_last_send_time   = 0.0
_prev_yaw         = 0.0
_prev_pitch       = 0.0
_lock             = threading.Lock()

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def set_tracking(enabled: bool):
    global _tracking_enabled
    _tracking_enabled = enabled

def is_tracking() -> bool:
    return _tracking_enabled

def process_frame(frame: np.ndarray, auto_send: bool = True):
    global _prev_yaw, _prev_pitch, _last_send_time

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (180, 180, 180), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (180, 180, 180), 1)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        cv2.putText(frame, "No Face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        return frame, [], None

    # 一番大きい顔
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_cx = x + fw // 2
    face_cy = y + fh // 2

    # 描画
    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (80, 180, 80), 2)
    cv2.circle(frame, (face_cx, face_cy), 4, (80, 180, 80), -1)

    # 画面中心からのズレを表示
    dx_disp = face_cx - cx
    dy_disp = face_cy - cy
    cv2.putText(frame, f"dx:{dx_disp} dy:{dy_disp}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 80), 1)

    if not _tracking_enabled:
        return frame, list(faces), None

    # 送信間隔チェック
    now = time.time()
    if now - _last_send_time < config.SEND_INTERVAL:
        return frame, list(faces), None

    # 角度計算
    dx = face_cx - cx
    dy = face_cy - cy

    # デッドゾーン
    if abs(dx) < config.DEAD_ZONE: dx = 0
    if abs(dy) < config.DEAD_ZONE: dy = 0

    # raw値に変換
    raw_yaw   = -(dx / (w / 2)) * config.HEAD_Y_MAX
    raw_pitch =  (dy / (h / 2)) * config.HEAD_P_MAX

    with _lock:
        # 最小変化量チェック
        if abs(raw_yaw - _prev_yaw) < config.MIN_ANGLE_CHANGE and \
           abs(raw_pitch - _prev_pitch) < config.MIN_ANGLE_CHANGE:
            return frame, list(faces), None
        
        angles = {"yaw": int(round(raw_yaw)), "pitch": int(round(raw_pitch))}

        if auto_send:
            controller.send(servo={
                "Head_Y": int(round(raw_yaw)),
                "Head_P": int(round(raw_pitch)),
            })

        _prev_yaw       = raw_yaw
        _prev_pitch     = raw_pitch
        _last_send_time = now

    return frame, list(faces), angles