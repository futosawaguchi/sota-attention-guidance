import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

def _download_model():
    if not os.path.exists(config.FACE_MODEL_PATH):
        print("[FaceDetector] モデルをダウンロード中...")
        urllib.request.urlretrieve(config.FACE_MODEL_URL, config.FACE_MODEL_PATH)
        print("[FaceDetector] ダウンロード完了")

class FaceDetector:
    def __init__(self):
        _download_model()
        self._latest_result = None
        base_options = python.BaseOptions(
            model_asset_path=config.FACE_MODEL_PATH
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
            output_facial_transformation_matrixes=True,
            result_callback=self._result_callback,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp  = 0

    def _result_callback(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def process(self, frame: np.ndarray) -> dict | None:
        """
        フレームを処理して顔の向きを返す
        Returns:
            {"yaw": float, "pitch": float, "roll": float} or None
        """
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 1
        self._landmarker.detect_async(mp_image, self._timestamp)

        if not self._latest_result:
            return None
        if not self._latest_result.face_landmarks:
            return None
        if not self._latest_result.facial_transformation_matrixes:
            return None

        return self._estimate_orientation(
            self._latest_result.facial_transformation_matrixes[0]
        )

    def _estimate_orientation(self, matrix) -> dict:
        mat   = np.array(matrix.data).reshape(4, 4)
        yaw   = float(np.degrees(np.arctan2(mat[2][0], mat[2][2])))
        pitch = float(np.degrees(np.arcsin(-mat[2][1])))
        roll  = float(np.degrees(np.arctan2(mat[0][1], mat[1][1])))
        return {
            "yaw":   round(yaw,   1),
            "pitch": round(pitch, 1),
            "roll":  round(roll,  1),
        }

    def get_latest_yaw(self) -> float:
        """最新のyaw角度を返す（結果がなければ0.0）"""
        if not self._latest_result:
            return 0.0
        if not self._latest_result.face_landmarks:
            return 0.0
        if not self._latest_result.facial_transformation_matrixes:
            return 0.0
        orientation = self._estimate_orientation(
            self._latest_result.facial_transformation_matrixes[0]
        )
        return orientation["yaw"]

    def close(self):
        self._landmarker.close()