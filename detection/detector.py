import cv2
import numpy as np
from ultralytics import YOLO
import config


class Detector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.confidence = config.YOLO_CONFIDENCE
        self.last_labels = set()

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], bool]:
        """
        フレームに対して物体検出を実行する

        Returns:
            annotated_frame: バウンディングボックス描画済みフレーム
            detections: 検出結果リスト [{"label": str, "confidence": float}]
            changed: 前回から検出物体の種類が変化したかどうか
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        result = results[0]

        detections = []
        current_labels = set()

        for box in result.boxes:
            label = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            current_labels.add(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({
                "label":      label,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2],
                "center":     [cx, cy],
            })

        # 検出物体の種類が変化したか判定
        changed = current_labels != self.last_labels
        self.last_labels = current_labels

        # バウンディングボックスを描画
        annotated_frame = self._draw(frame, result)

        return annotated_frame, detections, changed

    def _draw(self, frame: np.ndarray, result) -> np.ndarray:
        """バウンディングボックス・ラベル・信頼度を描画"""
        annotated = frame.copy()

        for box in result.boxes:
            label = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ボックス描画
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ラベル背景
            text = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)

            # ラベルテキスト
            cv2.putText(annotated, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated