import cv2
import time
import threading
import numpy as np
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import config


class AzureClient:
    def __init__(self):
        self.client = ImageAnalysisClient(
            endpoint=config.AZURE_CV_ENDPOINT,
            credential=AzureKeyCredential(config.AZURE_CV_API_KEY),
        )
        self.last_called = 0.0
        self.latest_result: dict | None = None
        self.lock = threading.Lock()
        self._analyzing = False

    def analyze_async(self, frame: np.ndarray) -> None:
        """
        クールダウン付きで非同期にAzure APIを呼び出す
        別スレッドで実行するので映像をブロックしない
        """
        now = time.time()
        if now - self.last_called < config.AZURE_COOLDOWN_SEC:
            return
        if self._analyzing:
            return

        self.last_called = now
        thread = threading.Thread(
            target=self._analyze, args=(frame.copy(),), daemon=True
        )
        thread.start()

    def _analyze(self, frame: np.ndarray) -> None:
        """Azure APIを呼び出して結果を保存する（内部用）"""
        self._analyzing = True
        try:
            # フレームをJPEGバイト列に変換
            _, buffer = cv2.imencode(".jpg", frame)
            image_data = buffer.tobytes()

            result = self.client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                ],
            )

            parsed = self._parse(result)
            with self.lock:
                self.latest_result = parsed

        except Exception as e:
            print(f"[Azure] エラー: {e}")
        finally:
            self._analyzing = False

    def _parse(self, result) -> dict:
        """Azure APIのレスポンスを扱いやすい形に整形する"""
        caption = ""
        if result.caption:
            caption = (
                f"{result.caption.text} "
                f"(信頼度: {result.caption.confidence:.2f})"
            )

        tags = [
            {"name": tag.name, "confidence": round(tag.confidence, 2)}
            for tag in (result.tags.list if result.tags else [])
        ]

        objects = [
            {"label": obj.tags[0].name, "confidence": round(obj.tags[0].confidence, 2)}
            for obj in (result.objects.list if result.objects else [])
            if obj.tags
        ]

        return {
            "caption": caption,
            "tags": tags,
            "objects": objects,
        }

    def get_latest_result(self) -> dict | None:
        """最新のAzure分析結果を返す"""
        with self.lock:
            return self.latest_result