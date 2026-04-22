import time
import threading
import math
import numpy as np
import config
from sota import controller as sota
from tracking import face_tracker
from voice import assistant

# ========== 状態定義 ==========
STATE_IDLE    = "idle"
STATE_GUIDING = "guiding"
STATE_SUCCESS = "success"

# ========== 設定 ==========
GUIDE_TIMEOUT_SEC   = 10.0   # 誘導タイムアウト（秒）
SUCCESS_HOLD_SEC    =  2.0   # 成功後の待機時間（秒）
FACE_ANGLE_THRESH   = 40.0   # ユーザが「見た」とみなす顔向き誤差（度）
CHECK_INTERVAL_SEC  =  0.2   # 判定ループの間隔（秒）

# ========== 内部状態 ==========
_state        = STATE_IDLE
_state_lock   = threading.Lock()
_target       = None   # 現在の注目対象 {"label", "center", "bbox"}
_guide_start  = 0.0
_running      = False

# ========== カメラ画角設定（キャリブレーション前の仮値） ==========
# カメラBの水平画角（度）※実際のカメラに合わせて調整
CAM_B_FOV_H = 60.0
# カメラBの垂直画角（度）
CAM_B_FOV_V = 45.0
# カメラBの解像度（detector.pyで使うものと合わせる）
CAM_B_W = 640
CAM_B_H = 480

# ========== 座標変換 ==========
def image_to_angle(center_x: int, center_y: int) -> tuple[float, float]:
    """
    カメラBの画像座標(cx, cy)からSotaの頭部・腕の目標角度を計算する

    画像中心を0度として、左右・上下の角度を返す
    Returns:
        yaw   : 左右角度（Sota座標系）正=右
        pitch : 上下角度（Sota座標系）正=下
    """
    # 画像中心からの相対位置（-0.5 〜 +0.5）
    rel_x = (center_x - CAM_B_W / 2) / CAM_B_W
    rel_y = (center_y - CAM_B_H / 2) / CAM_B_H

    # 画角から実際の角度に変換
    yaw   =  rel_x * CAM_B_FOV_H   # 右が正
    pitch =  rel_y * CAM_B_FOV_V   # 下が正

    # Sotaのサーボ値スケールに変換
    # Head_Y: -1400〜1400、Head_P: -290〜110
    sota_yaw   = yaw   * (1400 / (CAM_B_FOV_H / 2))
    sota_pitch = pitch * ( 100 / (CAM_B_FOV_V / 2))

    return sota_yaw, sota_pitch


def angle_to_arm(sota_yaw: float, sota_pitch: float) -> dict:
    """
    頭部の向き角度から腕のサーボ値を計算する
    右方向ならRShoulder、左方向ならLShoulderを使う

    Returns:
        サーボ値の辞書
    """
    if sota_yaw >= 0:
        # 右方向：右腕を上げる
        # RShoulder_P: -1400(前)〜800(後ろ) → 前に出すほどマイナス
        shoulder = max(-1400, min(800, -int(abs(sota_yaw) * 0.6)))
        elbow    = max( -900, min(650, -int(abs(sota_pitch) * 2.0)))
        return {
            "RShoulder_P": shoulder,
            "RElbow_P":    elbow,
            "LShoulder_P": 900,   # 左腕は初期位置
            "LElbow_P":    0,
        }
    else:
        # 左方向：左腕を上げる
        shoulder = max(-800, min(1400, int(abs(sota_yaw) * 0.6)))
        elbow    = max( -650, min(900, int(abs(sota_pitch) * 2.0)))
        return {
            "LShoulder_P": shoulder,
            "LElbow_P":    elbow,
            "RShoulder_P": -900,  # 右腕は初期位置
            "RElbow_P":    0,
        }


# ========== 誘導動作 ==========
def _do_guide(target: dict):
    """Sotaが物体方向へ頭・腕を向けて発話する"""
    cx, cy = target["center"]
    sota_yaw, sota_pitch = image_to_angle(cx, cy)
    arm_servo = angle_to_arm(sota_yaw, sota_pitch)

    # 頭を向ける
    sota.send(servo={
        "Head_Y": int(sota_yaw),
        "Head_P": int(sota_pitch),
        **arm_servo,
    })

    # 少し待ってから発話（動作と発話が重ならないように）
    time.sleep(0.5)
    assistant.play_tts(f"{target['label']}を見てください")


def _do_success():
    """誘導成功時の動作"""
    sota.send(servo={"Head_Y": 0, "Head_P": 0})
    time.sleep(0.3)
    assistant.play_tts("そうです、ありがとうございます")
    time.sleep(SUCCESS_HOLD_SEC)
    sota.reset_posture()


# ========== ユーザ視線判定 ==========
def _user_is_looking(target: dict, faces: list) -> bool:
    """
    ユーザが対象物体の方向を向いているか判定する

    face_tracker.process_frame()が返すanglesではなく
    顔の画像座標から簡易的に判定する
    """
    if not faces:
        return False

    # 最大の顔を使う
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_cx = x + fw // 2

    # カメラAの画像中心からのズレ（ピクセル）
    # カメラAの解像度（仮値）
    cam_a_w = 640
    dx = face_cx - cam_a_w // 2

    # ピクセルずれを角度に換算（カメラAの画角を60度と仮定）
    face_angle = (dx / (cam_a_w / 2)) * 30.0   # -30〜+30度

    # 物体のSota基準角度
    cx, cy = target["center"]
    sota_yaw, _ = image_to_angle(cx, cy)
    # SotaサーボからDeg換算
    target_angle = sota_yaw / (1400 / 30.0)

    diff = abs(face_angle - target_angle)
    return diff < FACE_ANGLE_THRESH


# ========== メインループ ==========
def _control_loop(get_detections, get_faces):
    """
    注意誘導のメインループ

    Args:
        get_detections: 最新の検出結果を返す関数 → list[dict]
        get_faces:      最新の顔リストを返す関数  → list
    """
    global _state, _target, _guide_start

    while _running:
        time.sleep(CHECK_INTERVAL_SEC)
        detections = get_detections()
        faces      = get_faces()

        with _state_lock:
            current = _state

        if current == STATE_IDLE:
            # 注目対象があれば誘導開始
            if detections:
                target = max(detections, key=lambda d: d["confidence"])
                with _state_lock:
                    _state       = STATE_GUIDING
                    _target      = target
                    _guide_start = time.time()
                threading.Thread(target=_do_guide,
                                 args=(target,), daemon=True).start()

        elif current == STATE_GUIDING:
            # タイムアウト判定
            if time.time() - _guide_start > GUIDE_TIMEOUT_SEC:
                with _state_lock:
                    _state = STATE_IDLE
                sota.reset_posture()
                continue

            # 成功判定
            if _target and _user_is_looking(_target, faces):
                with _state_lock:
                    _state = STATE_SUCCESS
                threading.Thread(target=_do_success, daemon=True).start()

        elif current == STATE_SUCCESS:
            # _do_success内でリセット済み、IDLEに戻す
            with _state_lock:
                _state  = STATE_IDLE
                _target = None


# ========== 外部インターフェース ==========
def start(get_detections, get_faces):
    """統合コントローラを起動する"""
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