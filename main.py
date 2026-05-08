import threading
import time
import cv2
from flask import Flask, Response, render_template, jsonify, request
import config
from sota import controller
from tracking import face_tracker
from detection.camera import Camera
from detection.detector import Detector
from detection.azure_client import AzureClient
from voice import assistant
from controller import attention_controller
import socket as _socket
from tracking.hand_detector import HandDetector

# 遅延初期化
_hand_detector_instance = None
_hand_detector_lock     = threading.Lock()

def get_hand_detector():
    global _hand_detector_instance
    with _hand_detector_lock:
        if _hand_detector_instance is None:
            print("[Main] HandDetector初期化中...")
            _hand_detector_instance = HandDetector()
            print("[Main] HandDetector初期化完了")
        return _hand_detector_instance

app = Flask(__name__)

# ========== モジュール初期化 ==========
camera_user = Camera(config.CAMERA_USER_INDEX)  # カメラA: ユーザ向け
camera_env  = Camera(config.CAMERA_ENV_INDEX)   # カメラB: 環境向け
detector    = Detector()
azure_client = AzureClient()

# ========== 共有データ ==========
_latest_user_frame = None
_latest_env_frame  = None
_latest_detections = []
_user_frame_lock   = threading.Lock()
_env_frame_lock    = threading.Lock()
_detections_lock   = threading.Lock()
_latest_face_angle = 0  # ユーザの顔角度（Head_Y値）
_face_angle_lock   = threading.Lock()

# ========== 共有データへのアクセサ関数 ==========
# attention_controllerに渡すゲッター関数
def get_latest_detections():
    with _detections_lock:
        return _latest_detections.copy()

# face_trackerから顔リストを取得するための変数を追加
_latest_faces = []
_faces_lock   = threading.Lock()

_latest_pointing   = None
_pointing_lock     = threading.Lock()
HAND_DETECT_INTERVAL = 1.0

# ========== カメラAループ（ユーザ顔追従） ==========
def camera_user_loop():
    global _latest_user_frame, _latest_faces, _latest_face_angle, _latest_pointing
    camera_user.start()
    last_hand_detect = 0.0
    while True:
        frame = camera_user.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        is_guiding = attention_controller.get_state() != "idle"
        
        # 誘導中は顔追従を完全に止める
        if is_guiding:
            # 誘導中: サーボ送信は止めるがMediaPipe顔向き推定は継続
            face_tracker._face_detector.process(frame)
            with _user_frame_lock:
                _latest_user_frame = frame
            continue
        frame, faces, angles = face_tracker.process_frame(frame)
        # 手検出（1秒に1回）
        now = time.time()
        if now - last_hand_detect >= HAND_DETECT_INTERVAL:
            try:
                hd = get_hand_detector()
                _, gesture, direction = hd.process(frame)
                with _pointing_lock:
                    _latest_pointing = direction
            except Exception as e:
                print(f"[Main] 手検出エラー: {e}")
            last_hand_detect = now
        with _user_frame_lock:
            _latest_user_frame = frame
        with _faces_lock:
            _latest_faces = faces
        # 顔角度をHead_Y値として保存
        if angles is not None:
            with _face_angle_lock:
                _latest_face_angle = angles["yaw"]

def get_latest_faces():
    with _faces_lock:
        return _latest_faces.copy()

def get_latest_face_angle():
    with _face_angle_lock:
        return _latest_face_angle
    
def get_latest_pointing():
    with _pointing_lock:
        return _latest_pointing

# ========== カメラBループ（物体検出） ==========
YOLO_INTERVAL = 10.0  # 推論間隔（秒)

def camera_env_loop():
    global _latest_env_frame, _latest_detections
    camera_env.start()
    last_detect = 0.0

    while True:
        frame = camera_env.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_detect < YOLO_INTERVAL:
            time.sleep(0.01)
            continue

        last_detect = now
        annotated_frame, detections, changed = detector.detect(frame)

        with _detections_lock:
            _latest_detections = detections
        with _env_frame_lock:
            _latest_env_frame = annotated_frame

        if changed:
            azure_client.analyze_async(annotated_frame)

# ========== 音声ループ ==========
def voice_loop():
    assistant.vad_loop()

# ========== MJPEGストリーム生成 ==========
def generate_user_stream():
    while True:
        with _user_frame_lock:
            frame = _latest_user_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(1/30)

def generate_env_stream():
    while True:
        with _env_frame_lock:
            frame = _latest_env_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(1/30)
 
# ========== トリガー受信 ==========       
# トリガー受信用UDPソケット
_trigger_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
_trigger_sock.bind(("127.0.0.1", 19001))
_trigger_sock.settimeout(1.0)

def trigger_receiver_loop():
    """trigger_server.pyからのUDP通知を受け取るループ"""
    print("[Main] トリガー受信待機中...")
    while True:
        try:
            data, _ = _trigger_sock.recvfrom(1024)
            if data == b"TRIGGER":
                print("[Main] トリガー受信!")
                _handle_trigger()
        except _socket.timeout:
            continue
        except Exception as e:
            print(f"[Main] トリガー受信エラー: {e}")

def _handle_trigger():
    """トリガー受信時の処理"""
    # 待機中以外は無視
    if attention_controller.get_state() != "idle":
        print("[Main] 誘導中のためトリガー無視")
        return

    # 手の検出結果を確認
    pointing = get_latest_pointing()
    if pointing is None:
        print("[Main] 手が検出されていないためトリガー無視")
        from voice.assistant import send_tts
        send_tts("手で指差しながら言ってください。")
        return

    # 現在の物体検出結果を取得
    detections = get_latest_detections()

    print(f"[Main] ユーザ誘導開始: pointing={pointing}")
    attention_controller.start_user_guidance(pointing, detections)

# ========== Flask ルーティング ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/user')
def video_feed_user():
    return Response(generate_user_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/env')
def video_feed_env():
    return Response(generate_env_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def api_detections():
    with _detections_lock:
        data = _latest_detections.copy()
    return jsonify(data)

@app.route('/api/azure')
def api_azure():
    result = azure_client.get_latest_result()
    return jsonify(result or {})

@app.route('/api/tracking', methods=['POST'])
def api_tracking():
    data = request.get_json()
    enabled = data.get('enabled', True)
    face_tracker.set_tracking(enabled)
    return jsonify({"status": "ok", "tracking": enabled})

@app.route('/api/led', methods=['POST'])
def api_led():
    data = request.get_json()
    color = data.get('color', 'green')
    controller.send(led=color)
    return jsonify({"status": "ok", "color": color})

@app.route('/api/motion', methods=['POST'])
def api_motion():
    data = request.get_json()
    motion = data.get('motion', 'nod')
    threading.Thread(target=controller.send,
                     kwargs={"motion": motion}, daemon=True).start()
    return jsonify({"status": "ok", "motion": motion})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    controller.reset_posture()
    return jsonify({"status": "ok"})

@app.route('/api/state')
def api_state():
    return jsonify({
        "state":  attention_controller.get_state(),
        "target": attention_controller.get_target(),
    })

@app.route('/api/face_orientation')
def api_face_orientation():
    from tracking import face_tracker
    orientation = face_tracker._face_detector.get_latest_yaw()
    target = attention_controller.get_target()
    
    target_yaw = None
    diff       = None
    if target:
        from controller.attention_controller import image_to_servo_values, FACE_ANGLE_THRESH
        cx, cy     = target["center"]
        servos     = image_to_servo_values(cx, cy)
        target_yaw = round(servos["Head_Y"] / (1400 / 70.0), 1)
        diff       = round(abs(orientation - target_yaw), 1)

    return jsonify({
        "user_yaw":   round(orientation, 1),
        "target_yaw": target_yaw,
        "diff":       diff,
        "threshold":  15.0,
        "is_looking": diff < 15.0 if diff is not None else False,
    })

# ========== 起動 ==========
if __name__ == '__main__':
    threading.Thread(target=camera_user_loop, daemon=True).start()
    threading.Thread(target=camera_env_loop,  daemon=True).start()
    #threading.Thread(target=voice_loop,        daemon=True).start()
    threading.Thread(target=trigger_receiver_loop, daemon=True).start()
    
    attention_controller.start(get_latest_detections, get_latest_faces, get_latest_face_angle)

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False, threaded=True)