import time
import threading
import json
import math
import numpy as np
import config
from sota import controller as sota
from voice.assistant import send_tts

# ========== 状態定義 ==========
STATE_IDLE    = "idle"
STATE_GUIDING = "guiding"
STATE_SUCCESS = "success"

# ========== 設定 ==========
GUIDE_TIMEOUT_SEC  = 15.0
SUCCESS_HOLD_SEC   =  2.0
FACE_ANGLE_THRESH  = 40.0
CHECK_INTERVAL_SEC =  0.2
COOLDOWN_SEC       = 30.0

# ========== キャリブレーションデータ読み込み ==========
with open("angle_calibration.json", "r") as f:
    _calib_points = json.load(f)

# numpy配列に変換
_pts     = np.array([[p["img_x"], p["img_y"]] for p in _calib_points], dtype=float)
_SERVO_KEYS = ["Waist_Y","RShoulder_P","RElbow_P","LShoulder_P","LElbow_P","Head_Y","Head_P","Head_R"]
_vals    = np.array([[p[k] for k in _SERVO_KEYS] for p in _calib_points], dtype=float)

# ========== 内部状態 ==========
_state          = STATE_IDLE
_state_lock     = threading.Lock()
_target         = None
_last_guide_end = 0.0
_running        = False
_last_labels    = set()

# ========== 補間：画像座標 → サーボ値 ==========
def image_to_servo_values(img_x: float, img_y: float) -> dict:
    """
    実測キャリブレーションデータから逆距離加重補間でサーボ値を計算する
    """
    query = np.array([img_x, img_y], dtype=float)

    # 各ポイントとの距離を計算
    dists = np.linalg.norm(_pts - query, axis=1)

    # 完全一致（距離0）の場合はそのまま返す
    if np.min(dists) < 1e-6:
        idx = np.argmin(dists)
        return {k: int(_vals[idx][i]) for i, k in enumerate(_SERVO_KEYS)}

    # 逆距離加重（IDW）補間
    weights = 1.0 / (dists ** 2)
    weights /= weights.sum()

    interpolated = np.dot(weights, _vals)

    result = {}
    for i, k in enumerate(_SERVO_KEYS):
        limits = {
            "Waist_Y":     (-1200, 1200),
            "RShoulder_P": (-1400,  800),
            "RElbow_P":    ( -900,  650),
            "LShoulder_P": ( -800, 1400),
            "LElbow_P":    ( -650,  900),
            "Head_Y":      (-1400, 1400),
            "Head_P":      ( -290,  110),
            "Head_R":      ( -300,  350),
        }
        lo, hi = limits[k]
        result[k] = int(max(lo, min(hi, round(interpolated[i]))))

    print(f"[Interp] img=({img_x:.0f},{img_y:.0f}) "
          f"Head_Y={result['Head_Y']} Head_P={result['Head_P']} "
          f"LShoulder={result['LShoulder_P']} RShoulder={result['RShoulder_P']}")

    return result


# ========== 視線判定 ==========
def _user_is_looking(target: dict, faces: list) -> bool:
    if not faces:
        return False
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_cx = x + fw // 2
    cam_a_w = 640
    dx = face_cx - cam_a_w // 2
    face_angle = (dx / (cam_a_w / 2)) * 30.0

    cx, cy = target["center"]
    servos = image_to_servo_values(cx, cy)
    target_angle = servos["Head_Y"] / (1400 / 30.0)

    return abs(face_angle - target_angle) < FACE_ANGLE_THRESH


# ========== クールダウン判定 ==========
def _in_cooldown() -> bool:
    return time.time() - _last_guide_end < COOLDOWN_SEC


# ========== 誘導ループ ==========
def _guide_loop(target: dict, get_faces):
    global _state, _last_guide_end

    cx, cy  = target["center"]
    servos  = image_to_servo_values(cx, cy)

    # 腕だけのサーボ値（頭は初期位置）
    arm_only = {
        "Waist_Y":     servos["Waist_Y"],
        "RShoulder_P": servos["RShoulder_P"],
        "RElbow_P":    servos["RElbow_P"],
        "LShoulder_P": servos["LShoulder_P"],
        "LElbow_P":    servos["LElbow_P"],
        "Head_Y":      0,
        "Head_P":      0,
        "Head_R":      0,
    }

    # 頭+腕のサーボ値（物体方向）
    all_servos = {**servos}

    start_time = time.time()

    # ① 腕を物体方向へ + 発話
    sota.send(servo=arm_only)
    time.sleep(0.5)
    send_tts(f"{target['label']}を見てください")
    time.sleep(0.3)

    loop_count = 0
    while True:
        if time.time() - start_time > GUIDE_TIMEOUT_SEC:
            break

        faces = get_faces()
        if _user_is_looking(target, faces):
            with _state_lock:
                _state = STATE_SUCCESS
            _do_success(target)
            _last_guide_end = time.time()
            return

        # ② 顔を物体方向へ（腕はそのまま）
        sota.send(servo=all_servos)
        time.sleep(1.5)

        if time.time() - start_time > GUIDE_TIMEOUT_SEC:
            break

        faces = get_faces()
        if _user_is_looking(target, faces):
            with _state_lock:
                _state = STATE_SUCCESS
            _do_success(target)
            _last_guide_end = time.time()
            return

        # ③ ユーザの方へ顔を戻す（腕はそのまま）
        sota.send(servo={**arm_only, "Head_Y": 0, "Head_P": 0})
        time.sleep(0.5)

        # ④ 発話（2ループに1回）
        if loop_count % 2 == 0:
            send_tts("こちらです、見てください")
        time.sleep(0.5)

        loop_count += 1

    # タイムアウト
    with _state_lock:
        _state = STATE_IDLE
    _last_guide_end = time.time()
    sota.reset_posture()


def _do_success(target: dict):
    sota.send(servo={"Head_Y": 0, "Head_P": 0})
    time.sleep(0.3)
    send_tts(f"ありがとうございます、{target['label']}ですね")
    time.sleep(SUCCESS_HOLD_SEC)
    sota.reset_posture()
    with _state_lock:
        global _state
        _state = STATE_IDLE


# ========== メインループ ==========
def _control_loop(get_detections, get_faces):
    global _state, _target, _last_labels

    while _running:
        time.sleep(CHECK_INTERVAL_SEC)

        with _state_lock:
            current = _state

        if current != STATE_IDLE:
            continue

        if _in_cooldown():
            continue

        detections   = get_detections()
        current_labels = {d["label"] for d in detections}

        changed      = current_labels != _last_labels
        _last_labels = current_labels

        if changed and detections and current_labels:
            target = max(detections, key=lambda d: d["confidence"])
            with _state_lock:
                _state  = STATE_GUIDING
                _target = target
            threading.Thread(
                target=_guide_loop,
                args=(target, get_faces),
                daemon=True
            ).start()


# ========== 外部インターフェース ==========
def start(get_detections, get_faces):
    global _running
    _running = True
    threading.Thread(
        target=_control_loop,
        args=(get_detections, get_faces),
        daemon=True
    ).start()

def get_state() -> str:
    with _state_lock:
        return _state

def get_target() -> dict:
    with _state_lock:
        return _target