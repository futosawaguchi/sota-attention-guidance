import time
import threading
import json
import math
import numpy as np
import config
from sota import controller as sota
from voice.assistant import send_tts
from tracking import face_tracker
from sota.controller import smooth_send as sota_smooth

# ========== 状態定義 ==========
STATE_IDLE    = "idle"
STATE_GUIDING = "guiding"
STATE_SUCCESS = "success"
STATE_USER_GUIDING = "user_guiding"

# ========== 設定 ==========
GUIDE_TIMEOUT_SEC  = 30.0
SUCCESS_HOLD_SEC   =  2.0
FACE_ANGLE_THRESH  = 15.0
CHECK_INTERVAL_SEC =  0.2
COOLDOWN_SEC       = 10.0

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
_guided_labels  = set()   # 誘導済みラベル
_guided_lock    = threading.Lock()
_all_guided    = False  # 全誘導完了フラグ
_reset_time    = 0.0    # リセット予定時刻

# ========== ラベル日本語変換 ==========
LABEL_JA = {
    "bottle":      "ボトル",
    "cup":         "カップ",
    "book":        "本",
    "cell phone":  "スマートフォン",
    "laptop":      "パソコン",
    "keyboard":    "キーボード",
    "mouse":       "マウス",
    "scissors":    "ハサミ",
    "pen":         "ペン",
    "clock":       "時計",
    "vase":        "花瓶",
    "bowl":        "ボウル",
    "banana":      "バナナ",
    "apple":       "リンゴ",
    "orange":      "オレンジ",
    "sandwich":    "サンドイッチ",
    "chair":       "椅子",
    "remote":      "リモコン",
    "backpack":    "バッグ",
    "umbrella":    "傘",
    "tape":        "テープ",
}

import socket as _socket

_speaking_notify_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)

def _notify_trigger_server(is_speaking: bool):
    """trigger_serverに発話中フラグを通知する"""
    msg = b"SPEAKING_START" if is_speaking else b"SPEAKING_END"
    _speaking_notify_sock.sendto(msg, ("127.0.0.1", 19002))

def _label_to_ja(label: str) -> str:
    """YOLOラベルを日本語に変換する（未登録はそのまま返す）"""
    return LABEL_JA.get(label, label)

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
def _user_is_looking(target: dict) -> bool:
    # MediaPipe FaceDetectorからyaw角度を取得
    user_yaw = face_tracker._face_detector.get_latest_yaw()

    cx, cy      = target["center"]
    servos      = image_to_servo_values(cx, cy)
    sota_head_y = servos["Head_Y"]
    target_yaw  = sota_head_y / (1400 / 70.0)

    diff = abs(user_yaw - target_yaw)
    print(f"[Looking] user_yaw={user_yaw:.1f} target_yaw={target_yaw:.1f} diff={diff:.1f}")

    return diff < FACE_ANGLE_THRESH


# ========== クールダウン判定 ==========
def _in_cooldown() -> bool:
    return time.time() - _last_guide_end < COOLDOWN_SEC

# ========== 除外ラベル ==========
EXCLUDE_LABELS = {"person", "tv"}

# ========== 誘導ループ ==========
def _guide_loop(target: dict, get_faces, get_face_angle):
    global _state, _last_guide_end

    cx, cy  = target["center"]
    servos  = image_to_servo_values(cx, cy)

    # 腕のみ（頭は正面）
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

    # 頭+腕（物体方向）
    all_servos = {**servos}

    start_time = time.time()

    # ① 腕を物体方向へ + 発話
    sota.send(servo=arm_only)
    time.sleep(0.3)
    send_tts("これを見てください")
    time.sleep(0.3)
    user_head_y = get_face_angle()
    loop_count = 0
    while True:
        if time.time() - start_time > GUIDE_TIMEOUT_SEC:
            break

        # ② 顔を物体方向へ（腕はそのまま）2秒
        sota_smooth(
            start_servo={**arm_only, "Head_Y": 0, "Head_P": 0},
            end_servo=all_servos,
            duration_sec=1.0
        )
        time.sleep(2.0)

        if time.time() - start_time > GUIDE_TIMEOUT_SEC:
            break
        
        # ③ ユーザの方へ顔を向ける（カメラAの顔追従角度を使用）2秒
        sota_smooth(
            start_servo=all_servos,
            end_servo={**arm_only, "Head_Y": user_head_y, "Head_P": 0},
            duration_sec=1.0
        )
        time.sleep(2.0)

        # 成功確認
        faces = get_faces()
        if _user_is_looking(target):
            with _state_lock:
                _state = STATE_SUCCESS
            _do_success(target)
            _last_guide_end = time.time()
            return

        # ④ 発話（2ループに1回）
        if loop_count % 2 == 0:
            send_tts("これを見てください")

        loop_count += 1
    
    # タイムアウト処理の前に誘導済みリストに追加
    with _guided_lock:
        _guided_labels.add(target["label"])
    label_ja = _label_to_ja(target["label"])
    send_tts(f"こちらの{label_ja}を確認されませんでした。")

    # タイムアウト
    with _state_lock:
        _state = STATE_IDLE
    _last_guide_end = time.time()
    sota.reset_posture()
    _last_guide_end = time.time()


def _do_success(target: dict):
    sota_smooth(
        start_servo=sota._current_posture.copy(),
        end_servo={"Head_Y": 0, "Head_P": 0},
        duration_sec=1.0
    )
    time.sleep(0.3)
    label_ja = _label_to_ja(target["label"])
    send_tts(f"ありがとうございます。こちらが{label_ja}でした。")
    time.sleep(SUCCESS_HOLD_SEC)
    sota.reset_posture()
    
    # 成功時も誘導済みリストに追加
    with _guided_lock:
        _guided_labels.add(target["label"])
    with _state_lock:
        global _state
        _state = STATE_IDLE


# ========== メインループ ==========
def _control_loop(get_detections, get_faces, get_face_angle):
    global _state, _target, _last_labels, _guided_labels
    global _all_guided, _reset_time

    while _running:
        time.sleep(CHECK_INTERVAL_SEC)

        with _state_lock:
            current = _state

        if current != STATE_IDLE:
            continue

        if _in_cooldown():
            continue

        # リセット待機中
        if _all_guided:
            if time.time() < _reset_time:
                continue
            # 15秒経過 → リセットして再開
            send_tts("もう一度紹介します。")
            with _guided_lock:
                _guided_labels = set()
            _all_guided = False
            continue

        detections     = get_detections()
        detections     = [d for d in detections if d["label"] not in EXCLUDE_LABELS]
        current_labels = {d["label"] for d in detections}

        # 新しいラベルが追加されたときだけ誘導済みリストから外す
        new_labels   = current_labels - _last_labels
        _last_labels = current_labels

        if new_labels:
            with _guided_lock:
                _guided_labels -= new_labels

        # 未誘導の物体だけ対象にする
        with _guided_lock:
            guided = _guided_labels.copy()

        candidates = [d for d in detections if d["label"] not in guided]

        if not candidates:
            # 検出物体が0件の場合は何もしない
            if not detections:
                continue
            # 全物体の誘導完了
            if not _all_guided:
                _all_guided  = True
                _reset_time  = time.time() + 15.0
                send_tts("全ての物体の紹介が完了しました。15秒後にもう一度紹介します。")
            continue

        # confidence最高のものを選ぶ
        target = max(candidates, key=lambda d: d["confidence"])

        with _state_lock:
            _state  = STATE_GUIDING
            _target = target

        threading.Thread(
            target=_guide_loop,
            args=(target, get_faces, get_face_angle),
            daemon=True
        ).start()


# ========== 外部インターフェース ==========
def start(get_detections, get_faces, get_face_angle):
    global _running
    _running = True
    threading.Thread(
        target=_control_loop,
        args=(get_detections, get_faces, get_face_angle),
        daemon=True
    ).start()

def get_state() -> str:
    with _state_lock:
        return _state

def get_target() -> dict:
    with _state_lock:
        return _target
    
def start_user_guidance(pointing_direction: dict, detections: list):
    """
    ユーザからの誘導を開始する
    main.pyから呼ばれる

    Parameters
    ----------
    pointing_direction: hand_detectorから取得した指の方向
    detections: 現在の物体検出結果
    """
    global _state

    with _state_lock:
        if _state != STATE_IDLE:
            return False  # 待機中以外は受け付けない
        _state = STATE_USER_GUIDING

    threading.Thread(
        target=_user_guide_loop,
        args=(pointing_direction, detections),
        daemon=True
    ).start()
    return True


def _user_guide_loop(pointing_direction: dict, detections: list):
    """ユーザからの誘導ループ"""
    global _state, _last_guide_end
    _notify_trigger_server(True)
    USER_GUIDE_LOOK_SEC = 5.0   # 物体を見る時間
    USER_GUIDE_TIMEOUT  = 20.0  # タイムアウト

    start_time = time.time()

    # 指の方向から最も近い物体を特定
    target = _find_target_from_pointing(pointing_direction, detections)

    if target is None:
        send_tts("指差している物体が見つかりませんでした。")
        _notify_trigger_server(False)
        with _state_lock:
            _state = STATE_IDLE
        return

    label_ja = _label_to_ja(target["label"])

    # Sotaが物体の方向へ向く
    cx, cy  = target["center"]
    servos  = image_to_servo_values(cx, cy)
    sota_smooth(
        start_servo=sota._current_posture.copy(),
        end_servo=servos,
        duration_sec=1.0
    )

    send_tts(f"{label_ja}を見ています。")
    time.sleep(USER_GUIDE_LOOK_SEC)

    # 正面に戻る
    sota_smooth(
        start_servo=sota._current_posture.copy(),
        end_servo={"Head_Y": 0, "Head_P": 0},
        duration_sec=1.0
    )
    send_tts(f"{label_ja}を見ました。")
    time.sleep(1.0)
    sota.reset_posture()
    _last_guide_end = time.time()
    _notify_trigger_server(False)
    with _state_lock:
        _state = STATE_IDLE


def _find_target_from_pointing(pointing_direction: dict, detections: list) -> dict | None:
    """
    指の水平角度から最も近い物体を特定する
    カメラAとカメラBの座標系が逆なので反転して計算する
    """
    if not detections:
        return None

    horiz        = pointing_direction["horizontal"]
    sota_head_y  = sota._current_posture.get("Head_Y", 0)
    sota_head_deg = sota_head_y / (1400 / 70.0)

    # カメラAの座標系を反転してSota基準角度に変換
    sota_based_angle = -horiz + sota_head_deg

    best_target = None
    best_diff   = float("inf")

    for d in detections:
        if d["label"] in EXCLUDE_LABELS:
            continue
        cx, cy    = d["center"]
        servos    = image_to_servo_values(cx, cy)
        obj_angle = servos["Head_Y"] / (1400 / 70.0)
        diff      = abs(sota_based_angle - obj_angle)
        if diff < best_diff:
            best_diff   = diff
            best_target = d

    # 30度以上離れていたら「指差していない」とみなす
    if best_diff > 30.0:
        return None

    print(f"[Pointing] sota_based={sota_based_angle:.1f} "
          f"target={best_target['label']} diff={best_diff:.1f}")

    return best_target