import os
from dotenv import load_dotenv

load_dotenv()

# ========== Sota ==========
SOTA_IP   = os.getenv("SOTA_IP", "192.168.11.5")
SOTA_PORT = int(os.getenv("SOTA_PORT", 9980))

# ========== カメラ ==========
CAMERA_USER_INDEX = int(os.getenv("CAMERA_USER_INDEX", 0))  # カメラA: ユーザ向け（顔追従）
CAMERA_ENV_INDEX  = int(os.getenv("CAMERA_ENV_INDEX",  1))  # カメラB: 環境向け（物体検出）

# ========== フェイストラッキング ==========
SEND_INTERVAL    = 0.4
DEAD_ZONE        = 30
SMOOTHING_ALPHA  = 0.0
MIN_ANGLE_CHANGE = 5.0
HEAD_Y_MAX       = 300.0
HEAD_P_MAX       = 100.0

# ========== YOLOv8 ==========
YOLO_MODEL      = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5

# ========== Azure Computer Vision ==========
AZURE_CV_ENDPOINT   = os.getenv("AZURE_CV_ENDPOINT")
AZURE_CV_API_KEY    = os.getenv("AZURE_CV_API_KEY")
AZURE_COOLDOWN_SEC  = 3

# ========== Azure OpenAI (音声) ==========
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_API_KEY")
AZURE_BASE_URL        = os.getenv("AZURE_BASE_URL")
AZURE_STT_DEPLOY      = os.getenv("AZURE_STT_DEPLOY",  "gpt-4o-transcribe-test")
AZURE_CHAT_DEPLOY     = os.getenv("AZURE_CHAT_DEPLOY", "gpt-5.2-chat-test")
AZURE_TTS_DEPLOY      = os.getenv("AZURE_TTS_DEPLOY",  "gpt-4o-mini-tts-test")

# ========== 机・空間設定（キャリブレーション用） ==========
TABLE_HEIGHT_M = float(os.getenv("TABLE_HEIGHT_M", 0.72))  # 机の高さ(m)
SOTA_HEIGHT_M  = float(os.getenv("SOTA_HEIGHT_M",  0.40))  # Sotaの高さ(m)

# ========== Flask ==========
FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))