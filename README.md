# sota-attention-guidance

Sota ロボットを用いたリアルタイム注意誘導システムです（共同注意を実現するための注意誘導）。  
物体検出・顔追従・音声対話・ロボット制御を統合し、Sota がユーザの注意を机上の物体へ誘導します。

---

## システム概要

```
カメラA（Sota上搭載）
  └─ OpenCV 顔検出 → Sota 頭部追従（UDP）
  └─ MediaPipe FaceLandmarker → ユーザ顔向き推定（Yaw/Pitch/Roll）

カメラB（環境向け・三脚固定）
  └─ YOLOv8 → リアルタイム物体検出
  └─ Azure Computer Vision → 詳細画像分析

音声システム（独立プロセス）
  └─ Azure Whisper STT → 音声認識
  └─ Azure GPT → 応答生成
  └─ Azure TTS → 音声合成（別プロセスで音声競合を回避）

統合コントローラ（attention_controller.py）
  └─ 物体検出 → Sota が腕・頭で物体を指示
  └─ ユーザ顔向き判定 → 注意誘導成功を検出
  └─ ステートマシン（IDLE / GUIDING / SUCCESS）
```

---

## 機能

- **リアルタイム顔追従**：カメラAでユーザの顔を検出し、Sota の頭部がユーザを向き続ける
- **物体検出**：カメラBでYOLOv8による常時検出とAzure Computer Visionによる詳細分析
- **注意誘導**：新しい物体が検出されると、Sota が腕で指差し・頭部を物体方向へ向けて誘導
- **顔向き推定**：MediaPipe FaceLandmarkerでユーザのYaw/Pitch/Rollをリアルタイム推定
- **音声対話**：ユーザとの音声会話（別プロセスで音声競合を解消）
- **WebUI**：カメラ映像・検出結果・誘導状態をブラウザでリアルタイム確認
- **スムーズ動作**：Ease in-out補間でSotaのサーボを滑らかに制御（実装予定）

---

## ディレクトリ構成

```
sota-attention-guidance/
├── main.py                        # 統合エントリーポイント
├── tts_server.py                  # TTS専用プロセス（音声競合回避）
├── config.py                      # 全設定を一元管理
├── calibration.json               # カメラB位置キャリブレーション
├── angle_calibration.json         # サーボ角度キャリブレーション（11点）
├── calibrate_camera.py            # カメラキャリブレーションツール
├── calibrate_angles.py            # サーボ角度キャリブレーションツール
├── .env                           # 環境変数（Gitに含まれない）
├── .env.example                   # 環境変数テンプレート
├── requirements.txt
│
├── controller/
│   └── attention_controller.py   # 注意誘導の中枢ロジック・ステートマシン
│
├── sota/
│   └── controller.py             # Sota UDP制御・smooth_send
│
├── tracking/
│   ├── face_tracker.py           # OpenCV顔検出・頭部角度計算
│   ├── face_detector.py          # MediaPipe顔向き推定（Yaw/Pitch/Roll）
│   ├── hand_detector.py          # MediaPipe手検出・指差し方向推定
│   └── gestures/
│       ├── __init__.py
│       ├── base.py               # ジェスチャー基底クラス
│       └── pointing.py           # ポインティングジェスチャー検出
│
├── detection/
│   ├── camera.py                 # カメラ制御・フレーム取得
│   ├── detector.py               # YOLOv8物体検出
│   └── azure_client.py           # Azure Computer Vision連携
│
├── voice/
│   └── assistant.py              # 音声認識・GPT応答・TTS
│
├── templates/
│   └── index.html                # WebUI
├── static/
│   └── style.css                 # スタイルシート
└── java/
    └── SotaController.java       # Sota側Javaプログラム
```

---

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/futosawaguchi/sota-attention-guidance.git
cd sota-attention-guidance
```

### 2. 仮想環境の作成・有効化

```bash
python -m venv venv
source venv/bin/activate
```

### 3. パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env.example`をコピーして`.env`を作成してください。

```bash
cp .env.example .env
```

`.env`に以下を設定してください。

```
SOTA_IP=192.168.xx.xx
SOTA_PORT=9980

CAMERA_USER_INDEX=0
CAMERA_ENV_INDEX=1

AZURE_CV_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_CV_API_KEY=your_key_here

AZURE_API_KEY=your_key_here
AZURE_BASE_URL=https://your-resource.openai.azure.com/
AZURE_STT_DEPLOY=your_stt_deployment
AZURE_CHAT_DEPLOY=your_chat_deployment
AZURE_TTS_DEPLOY=your_tts_deployment

TABLE_HEIGHT_M=0.76
SOTA_HEIGHT_M=0.40

FLASK_PORT=5001
```

### 5. キャリブレーション

#### カメラBの位置登録

```bash
python calibrate_camera.py
```

カメラB映像が開くので、Sotaの頭部をクリックして`s`で保存してください。

#### サーボ角度キャリブレーション

```bash
python calibrate_angles.py
```

様々な位置に物体を置いてSotaの正しい向きを記録してください（最低5点推奨）。

### 6. 起動

**ターミナル1（TTS専用プロセス）:**

```bash
python tts_server.py
```

**ターミナル2（メインシステム）:**

```bash
python main.py
```

ブラウザで `http://localhost:5001` を開いてください。

---

## 物理配置

```
上から見た図:

        壁・モニター
    [Sota]（モニター前・壁際）
     正面↓（ユーザ方向）
   ─────────────── 机（高さ76cm）
   [物体エリア]（高さ78cm）
        \
    45度 \  60cm
          \
          📷カメラB（高さ115cm・三脚）
          👤ユーザ
```

| 項目 | 値 |
|---|---|
| カメラBの高さ | 115cm |
| Sota頭部の高さ | 104cm（机76cm + 28cm） |
| 物体エリアの高さ | 78cm（机76cm + 天板2cm） |
| カメラB〜Sota距離 | 60cm |
| カメラBの角度 | Sotaから見て右斜め前45度 |

---

## 注意誘導の動作フロー

```
カメラBで新しい物体を検出（changed=True）
        ↓
【STATE: GUIDING】
① Sotaが腕を物体方向へ向ける（スムーズに1.5秒）
   + 発話「これを見てください」
        ↓
ループ:
② Sotaが顔を物体方向へ向ける（スムーズに1.5秒）
   2秒待機
        ↓
③ Sotaがユーザの方へ顔を戻す（スムーズに1.5秒）
   2秒待機
        ↓
④ ユーザの顔向き（Yaw）を判定
   → 成功: STATE_SUCCESS → 「ありがとうございます」
   → 失敗: 発話「これを見てください」→ ②へ戻る
        ↓
タイムアウト（15秒）→ STATE_IDLE
クールダウン（30秒）→ 次の誘導へ
```

---

## キャリブレーションについて

本システムはカメラBの画像座標からSotaのサーボ角度を**逆距離加重補間（IDW）**で計算します。

事前に`calibrate_angles.py`で複数点（推奨11点以上）の対応関係を記録することで、数式ベースのキャリブレーションより高精度な制御が可能です。

```
カメラB画像上の物体座標(u, v)
        ↓
angle_calibration.json（11点の実測データ）
        ↓
IDW補間 → 全サーボ値（頭部・腕・腰）を同時に計算
        ↓
Sota に UDP で送信
```

---

## 使用技術

| カテゴリ | 技術 |
|---|---|
| ロボット制御 | Sota（Vstone）/ UDP通信 / Java |
| 物体検出 | YOLOv8n（Ultralytics） |
| 画像分析 | Azure Computer Vision |
| 顔検出・追従 | OpenCV Haar Cascade |
| 顔向き推定 | MediaPipe FaceLandmarker（Tasks API） |
| 手検出・指差し | MediaPipe HandLandmarker（Tasks API） |
| 音声認識 | Azure OpenAI Whisper（gpt-4o-transcribe） |
| AI応答 | Azure OpenAI GPT |
| 音声合成 | Azure OpenAI TTS（gpt-4o-mini-tts） |
| Webフレームワーク | Flask |
| 言語 | Python 3.12 |

---

## 発展予定

- **ユーザ→Sota注意誘導**：ユーザがポインティングジェスチャーで物体を指し、Sotaがその方向を見る（`hand_detector.py`統合済み）
- **双方向注意共有（Joint Attention）**：Sota←→ユーザの双方向注意誘導の実現
- **音声トリガー**：「これを見て」とユーザが発話したときに注意誘導を開始
