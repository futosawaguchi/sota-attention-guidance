import time
import threading
import config
from sota import controller as sota
#from voice import assistantd 
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

# ========== カメラB画角設定 ==========
CAM_B_FOV_H = 60.0
CAM_B_FOV_V = 45.0
CAM_B_W     = 640
CAM_B_H     = 480

# ========== 内部状態 ==========
_state         = STATE_IDLE
_state_lock    = threading.Lock()
_target        = None
_last_guide_end = 0.0
_running       = False
_last_labels   = set()

# ========== 座標変換 ==========
def image_to_sota_angles(cx: int, cy: int) -> tuple[int, int]:
    rel_x =  (cx - CAM_B_W / 2) / CAM_B_W
    rel_y =  (cy - CAM_B_H / 2) / CAM_B_H
    yaw   = int( rel_x * CAM_B_FOV_H * (1400 / (CAM_B_FOV_H / 2)))
    pitch = int( rel_y * CAM_B_FOV_V * ( 100 / (CAM_B_FOV_V / 2)))
    yaw   = max(-1400, min(1400, yaw))
    pitch = max( -290, min( 110, pitch))
    return yaw, pitch

def calc_arm(yaw: int) -> dict:
    """yaw方向に応じて腕サーボ値を計算"""
    if yaw >= 0:
        shoulder = max(-1400, min(800, -int(abs(yaw) * 0.6)))
        return {
            "RShoulder_P": shoulder,
            "RElbow_P":    -200,
            "LShoulder_P":  900,
            "LElbow_P":     0,
        }
    else:
        shoulder = max(-800, min(1400, int(abs(yaw) * 0.6)))
        return {
            "LShoulder_P": shoulder,
            "LElbow_P":     200,
            "RShoulder_P": -900,
            "RElbow_P":     0,
        }

# ========== 視線判定 ==========
def _user_is_looking(target: dict, faces: list) -> bool:
    if not faces:
        return False
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_cx  = x + fw // 2
    cam_a_w  = 640
    dx       = face_cx - cam_a_w // 2
    face_angle  = (dx / (cam_a_w / 2)) * 30.0
    cx, cy      = target["center"]
    sota_yaw, _ = image_to_sota_angles(cx, cy)
    target_angle = sota_yaw / (1400 / 30.0)
    return abs(face_angle - target_angle) < FACE_ANGLE_THRESH

# ========== クールダウン判定 ==========
def _in_cooldown() -> bool:
    return time.time() - _last_guide_end < COOLDOWN_SEC

# ========== 誘導ループ ==========
def _guide_loop(target: dict, get_faces):
    global _state, _last_guide_end

    cx, cy    = target["center"]
    yaw, pitch = image_to_sota_angles(cx, cy)
    arm        = calc_arm(yaw)

    start_time = time.time()

    # ① 腕を物体方向へ + 発話
    sota.send(servo={**arm})
    time.sleep(0.5)
    send_tts(f"{target['label']}を見てください")
    time.sleep(0.3)

    loop_count = 0
    while True:
        # タイムアウト確認
        if time.time() - start_time > GUIDE_TIMEOUT_SEC:
            break

        # 成功確認
        faces = get_faces()
        if _user_is_looking(target, faces):
            with _state_lock:
                _state = STATE_SUCCESS
            _do_success(target)
            _last_guide_end = time.time()
            return

        # ② 顔を物体方向へ（腕はそのまま）
        sota.send(servo={
            "Head_Y": yaw,
            "Head_P": pitch,
            **arm,
        })
        time.sleep(1.5)

        # タイムアウト・成功確認
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
        sota.send(servo={
            "Head_Y": 0,
            "Head_P": 0,
            **arm,
        })
        time.sleep(0.5)

        # ④ 発話（2ループに1回）
        if loop_count % 2 == 0:
            send_tts("こちらです、見てください")
        time.sleep(0.5)

        loop_count += 1

    # タイムアウト処理
    with _state_lock:
        _state = STATE_IDLE
    _last_guide_end = time.time()
    sota.reset_posture()

def _do_success(target: dict):
    """成功時の動作"""
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

        detections = get_detections()
        current_labels = {d["label"] for d in detections}

        # 新しい物体が来たときだけ発火
        changed = current_labels != _last_labels
        _last_labels = current_labels

        if changed and detections:
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