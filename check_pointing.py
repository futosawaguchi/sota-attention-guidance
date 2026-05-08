# check_pointing.py
import sys
import time
import threading
import cv2
import numpy as np
sys.path.insert(0, '.')

import config
from tracking.hand_detector import HandDetector
from detection.camera import Camera
from detection.detector import Detector
from controller.attention_controller import image_to_servo_values, EXCLUDE_LABELS
from sota import controller as sota

# ========== 初期化 ==========
hand_detector = HandDetector()
camera_a      = Camera(config.CAMERA_USER_INDEX)
camera_b      = Camera(config.CAMERA_ENV_INDEX)
detector      = Detector()

# ========== 共有データ ==========
_latest_detections = []
_det_lock          = threading.Lock()
_latest_pointing   = None
_pointing_lock     = threading.Lock()

def detection_loop():
    """カメラBで物体検出し続けるループ"""
    global _latest_detections
    camera_b.start()
    last_detect = 0.0
    while True:
        frame = camera_b.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        now = time.time()
        if now - last_detect < 0.5:
            time.sleep(0.01)
            continue
        last_detect = now
        _, detections, _ = detector.detect(frame)
        detections = [d for d in detections if d["label"] not in EXCLUDE_LABELS]
        with _det_lock:
            _latest_detections = detections

threading.Thread(target=detection_loop, daemon=True).start()

# ========== メイン ==========
camera_a.start()
time.sleep(1.0)  # カメラ起動待ち

print("check_pointing.py 起動")
print("カメラAに手を映してください")
print("'q'で終了")

while True:
    frame = camera_a.get_frame()
    if frame is None:
        time.sleep(0.01)
        continue

    # 手検出
    frame, gesture, direction = hand_detector.process(frame)

    # 現在のSotaのHead_Y（待機中なので0）
    sota_head_y = sota._current_posture.get("Head_Y", 0)

    h, w = frame.shape[:2]

    # 情報表示エリア（右側に黒背景で表示）
    info_w  = 400
    canvas  = np.zeros((h, w + info_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    cv2.rectangle(canvas, (w, 0), (w + info_w, h), (20, 20, 20), -1)

    y = 30
    def put(text, color=(200, 200, 200)):
        global y
        cv2.putText(canvas, text, (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 28

    put("=== POINTING DEBUG ===", (100, 200, 255))
    put("")

    if direction is not None:
        horiz      = direction["horizontal"]
        vert       = direction["vertical"]
        depth      = direction["depth"]

        # Sota基準角度 = カメラAの指角度 + SotaのHead_Y角度換算
        sota_head_deg  = sota_head_y / (1400 / 70.0)
        sota_based_angle = -horiz + sota_head_deg

        put(f"Gesture: {gesture or 'None'}", (100, 255, 100))
        put(f"Horizontal: {horiz:+.1f} deg")
        put(f"Vertical:   {vert:+.1f} deg")
        put(f"Depth:      {depth:+.1f} deg")
        put("")
        put(f"Sota Head_Y: {sota_head_y}", (200, 200, 100))
        put(f"Sota HeadDeg: {sota_head_deg:+.1f}", (200, 200, 100))
        put(f"Sota Based:  {sota_based_angle:+.1f} deg", (100, 255, 255))
        put("")
        put("=== OBJECTS ===", (100, 200, 255))

        with _det_lock:
            detections = _latest_detections.copy()

        if not detections:
            put("No objects detected", (150, 150, 150))
        else:
            best_target = None
            best_diff   = float("inf")

            for d in detections:
                cx, cy    = d["center"]
                servos    = image_to_servo_values(cx, cy)
                obj_angle = servos["Head_Y"] / (1400 / 70.0)
                diff      = abs(sota_based_angle - obj_angle)

                if diff < best_diff:
                    best_diff   = diff
                    best_target = d

                # 色: 最も近い物体は緑、それ以外は白
                color = (100, 255, 100) if d == best_target else (200, 200, 200)
                put(f"{d['label']:<12} obj:{obj_angle:+5.1f} diff:{diff:4.1f}", color)

            put("")
            if best_diff < 30.0:
                put(f"-> {best_target['label']}", (0, 255, 0))
                put(f"   diff={best_diff:.1f} deg", (0, 255, 0))
            else:
                put(f"-> None (min diff={best_diff:.1f})", (100, 100, 255))

    else:
        put("No hand detected", (150, 150, 150))
        put("")
        put("=== OBJECTS ===", (100, 200, 255))
        with _det_lock:
            detections = _latest_detections.copy()
        if not detections:
            put("No objects detected", (150, 150, 150))
        else:
            for d in detections:
                cx, cy    = d["center"]
                servos    = image_to_servo_values(cx, cy)
                obj_angle = servos["Head_Y"] / (1400 / 70.0)
                put(f"{d['label']:<12} obj:{obj_angle:+5.1f}", (200, 200, 200))

    y = 30  # リセット
    cv2.imshow("Pointing Debug", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_a.stop()
camera_b.stop()
cv2.destroyAllWindows()