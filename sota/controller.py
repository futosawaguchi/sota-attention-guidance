import json
import socket
from config import SOTA_IP, SOTA_PORT
import time
import threading

# ========== UDP ソケット初期化 ==========
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_serv_address = (SOTA_IP, SOTA_PORT)

_send_lock = threading.Lock()

# ========== 現在のサーボ値を保持 ==========
_current_posture = {
    "Waist_Y":      0,
    "RShoulder_P": -900,
    "RElbow_P":     0,
    "LShoulder_P":  900,
    "LElbow_P":     0,
    "Head_Y":       0,
    "Head_P":       0,
    "Head_R":       0,
}

def send(servo=None, led=None, motion=None):
    """
    SOTAにUDPでJSONを送信する

    Parameters
    ----------
    servo  : dict | None  例) {"Head_Y": 100, "Head_P": -50}
    led    : str  | None  例) "green" / "blue" / "white" / "red" / "off"
    motion : str  | None  例) "nod" / "bye_bye" / "shake_head"
                               "right_hand_up" / "left_hand_up" / "both_hands_up"
        
    Waist_Y      (腰 左右):   -1200 〜 1200
    RShoulder_P  (右肩 前後): -1400 〜  800
    RElbow_P     (右肘 前後):  -900 〜  650
    LShoulder_P  (左肩 前後):  -800 〜 1400
    LElbow_P     (左肘 前後):  -650 〜  900
    Head_Y       (首 左右):   -1400 〜 1400
    Head_P       (首 前後):    -290 〜  110
    Head_R       (首 傾き):    -300 〜  350
    """
    global _current_posture
    with _send_lock:
        pos = _current_posture.copy()

        if servo:
            pos.update(servo)
            _current_posture.update(servo)

        if led:
            pos["LED"] = led

        if motion:
            pos["Motion"] = motion

        try:
            _sock.sendto(json.dumps(pos).encode("utf-8"), _serv_address)
        except Exception as e:
            print(f"[controller] UDP送信エラー: {e}")

def reset_posture():
    """初期姿勢に戻す"""
    global _current_posture
    _current_posture = {
        "Waist_Y":      0,
        "RShoulder_P": -900,
        "RElbow_P":     0,
        "LShoulder_P":  900,
        "LElbow_P":     0,
        "Head_Y":       0,
        "Head_P":       0,
        "Head_R":       0,
    }
    send()
    
def smooth_send(
    start_servo: dict,
    end_servo:   dict,
    duration_sec: float,
    steps: int = 20
):
    """
    startからendまでをduration_sec秒かけてsteps回に分けて送信する
    イージング（ease in-out）で自然な動きにする

    Parameters
    ----------
    start_servo  : 開始サーボ値の辞書
    end_servo    : 目標サーボ値の辞書
    duration_sec : 移動にかける時間（秒）
    steps        : 分割数（多いほど滑らか・デフォルト20）
    """
    global _current_posture
    with _send_lock:
        interval = duration_sec / steps

        for i in range(1, steps + 1):
            # ease in-out: 最初と最後はゆっくり、中間は速く
            t_linear = i / steps
            t = t_linear * t_linear * (3 - 2 * t_linear)  # smoothstep

            interpolated = {}
            for key in end_servo:
                start_val = start_servo.get(key, _current_posture.get(key, 0))
                end_val   = end_servo[key]
                interpolated[key] = int(start_val + (end_val - start_val) * t)

            _sock.sendto(
                json.dumps(interpolated).encode("utf-8"),
                _serv_address
            )
            time.sleep(interval)

        # 最終値を_current_postureに反映
        _current_posture.update(end_servo)