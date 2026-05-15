# sota-attention-guidance

Sota ロボットを用いた双方向リアルタイム注意誘導システムです。  
物体検出・顔追従・音声対話・ロボット制御を統合し、**Sota→ユーザ**および**ユーザ→Sota**の双方向注意誘導（Joint Attention）を実現します。

---

## システム概要

```
カメラA（Sota上搭載・ユーザ方向）
  └─ OpenCV 顔検出 → Sota 頭部追従（UDP）
  └─ MediaPipe FaceLandmarker → ユーザ顔向き推定（Yaw/Pitch/Roll）
  └─ MediaPipe HandLandmarker → 指差し方向推定（水平・垂直・奥行き）

カメラB（環境向け・三脚固定）
  └─ YOLOv8 → リアルタイム物体検出
  └─ Azure Computer Vision → 詳細画像分析

音声システム（独立プロセス）
  └─ tts_server.py    : Azure TTS による音声合成・再生
  └─ trigger_server.py: Azure Whisper STT によるトリガーワード検出

統合コントローラ（attention_controller.py）
  └─ Sota→ユーザ誘導: 物体検出 → Sotaが腕・頭で物体を指示
  └─ ユーザ→Sota誘導: 指差し+音声 → Sotaが指定物体を確認
  └─ ステートマシン（IDLE / GUIDING / SUCCESS / USER_GUIDING）
```

---

## 機能

### Sota → ユーザ 注意誘導
- 新しい物体が検出されると自動的に誘導開始
- Sotaが腕で指差し＋頭部を物体方向へスムーズに向ける
- ユーザの顔向き（Yaw角度）で「見た」かどうかを判定
- 成功時・タイムアウト時に音声フィードバック
- 全物体の誘導完了後、15秒待機して再誘導

### ユーザ → Sota 注意誘導
- 「これを見てください」などのトリガーワード＋指差しで誘導
- MediaPipe手検出で指の方向を推定し、物体を特定
- Sotaが該当物体方向へスムーズに頭を向けて確認
- 「〇〇を見ています」→「〇〇を見ました」と音声フィードバック

### その他
- リアルタイム顔追従（待機中はSotaがユーザを向き続ける）
- WebUIでカメラ映像・検出結果・誘導状態をリアルタイム確認
- ユーザ顔向き（Yaw/Target/Diff）のリアルタイム表示
- Sota発話中はトリガー検出を自動停止（誤検出防止）

---

## ディレクトリ構成

```
sota-attention-guidance/
├── main.py                        # 統合エントリーポイント
├── tts_server.py                  # TTS専用プロセス（音声競合回避）
├── trigger_server.py              # トリガーワード検出専用プロセス
├── config.py                      # 全設定を一元管理
├── calibration.json               # カメラB位置キャリブレーション
├── angle_calibration.json         # サーボ角度キャリブレーション
├── calibrate_camera.py            # カメラキャリブレーションツール
├── calibrate_angles.py            # サーボ角度キャリブレーションツール
├── check_pointing.py              # 指差し方向デバッグツール
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
│   └── assistant.py              # TTS・STT共通関数
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
TTS_OUTPUT_DEVICE=5      # MacBook Proのスピーカー（sd.query_devices()で確認）
TRIGGER_MIC_DEVICE=1     # 使用するマイク（sd.query_devices()で確認）
```

> **Note**  
> デバイスインデックスはBluetoothデバイスの接続状態で変わることがあります。  
> 起動前に必ず `python -c "import sounddevice as sd; print(sd.query_devices())"` で確認してください。

### 5. キャリブレーション

#### カメラBの位置登録（初回のみ）

```bash
python calibrate_camera.py
```

カメラB映像が開くのでSotaの頭部をクリックして`s`で保存してください。

#### サーボ角度キャリブレーション（初回・配置変更時）

```bash
python calibrate_angles.py
```

様々な位置に物体を置いてSotaの正しい向きを記録してください（推奨11点以上）。

### 6. 起動

**3つのターミナルで起動してください。**

```bash
# ターミナル1: TTS専用プロセス
python tts_server.py

# ターミナル2: トリガー検出専用プロセス
python trigger_server.py

# ターミナル3: メインシステム
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

### Sota → ユーザ

```
カメラBで新しい物体を検出
        ↓
【STATE: GUIDING】
① Sotaが腕を物体方向へ（1.0秒でスムーズに移動）
   発話「これを見てください」
        ↓
ループ（最大30秒）:
② Sotaが顔を物体方向へ（1.0秒）→ 2秒待機
③ Sotaがユーザ方向へ顔を戻す（1.0秒）→ 2秒待機
④ ユーザの顔向きYaw角度で判定（閾値15度）
   → 成功: 「〇〇でした」→ クールダウン12秒
   → 失敗: 「これを見てください」→ ②へ
        ↓
タイムアウト（30秒）→「〇〇を確認されませんでした」
全物体完了 → 「全ての紹介が完了しました」→ 15秒後に再開
```

### ユーザ → Sota

```
ユーザが手で物体を指差しながら
「これを見てください」などと発話
        ↓
trigger_server.pyがトリガーワードを検出
        ↓
main.pyが手の検出結果を確認（2秒以内）
手なし → 「手で指差しながら言ってください」
        ↓
【STATE: USER_GUIDING】
指の水平角度からカメラBの物体を特定
        ↓
Sotaが物体方向へ頭をスムーズに向ける
発話「〇〇を見ています」→ 5秒待機
        ↓
正面に戻る → 発話「〇〇を見ました」
クールダウン12秒 → Sota誘導再開
```

---

## キャリブレーションについて

本システムはカメラBの画像座標からSotaのサーボ角度を**逆距離加重補間（IDW）**で計算します。

```
カメラB画像上の物体座標(u, v)
        ↓
angle_calibration.json（実測データ）
        ↓
IDW補間 → 全サーボ値（頭部・腕・腰）を同時に計算
        ↓
Sota に UDP で送信（smooth_send でスムーズに移動）
```

### 指差し方向の座標変換

カメラAとカメラBは座標系が異なるため、以下の変換を行います。

```
カメラA（Sota上）の指のhorizontal角度
        ↓
符号反転（カメラA・B座標系の違いを補正）
        ↓
+ Sotaの現在Head_Y角度（度換算）
        ↓
Sota基準の指差し方向角度
        ↓
カメラBの各物体のHead_Y角度と比較 → 最も近い物体を特定
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
| 音声合成 | Azure OpenAI TTS（gpt-4o-mini-tts） |
| Webフレームワーク | Flask |
| 言語 | Python 3.12 |

---

## トラブルシューティング

**音声が出ない**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
出力デバイスのインデックスを確認して`.env`の`TTS_OUTPUT_DEVICE`を更新してください。

**トリガーが検出されない**
`.env`の`TRIGGER_MIC_DEVICE`が正しいか確認してください。

**Sotaが正しい方向を向かない**
`calibrate_angles.py`でキャリブレーションポイントを追加してください。

**カメラが起動しない**
`.env`の`CAMERA_USER_INDEX`と`CAMERA_ENV_INDEX`が正しいか確認してください。

**指差し方向がずれる**
```bash
python check_pointing.py
```
でデバッグ画面を確認してください。
