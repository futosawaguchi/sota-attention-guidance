from abc import ABC, abstractmethod
import numpy as np


def _vec(a, b) -> np.ndarray:
    """ランドマークa → b のベクトル"""
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z])


def _angle(a, b, c) -> float:
    """
    3点a-b-cが作る角度（度）を返す
    b が頂点（関節）
    """
    v1 = _vec(b, a)
    v2 = _vec(b, c)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def get_finger_states(landmarks, threshold: float = 150.0) -> dict:
    """
    各指の関節角度から伸びているか曲がっているかを判定する

    判定方法：
    - 付け根・第2関節・指先の3点で角度を計算
    - threshold度以上 → 伸びている（True）
    - threshold度未満 → 曲がっている（False）

    threshold=150.0 は「ほぼまっすぐ」の基準
    """
    # 各指：(付け根, 第2関節, 指先)
    finger_joints = {
        "thumb":  (landmarks[1],  landmarks[3],  landmarks[4]),
        "index":  (landmarks[5],  landmarks[7],  landmarks[8]),
        "middle": (landmarks[9],  landmarks[11], landmarks[12]),
        "ring":   (landmarks[13], landmarks[15], landmarks[16]),
        "pinky":  (landmarks[17], landmarks[19], landmarks[20]),
    }

    result = {}
    for name, (base, mid, tip) in finger_joints.items():
        angle = _angle(base, mid, tip)
        result[name] = angle >= threshold

    return result


def thumb_near(landmarks, target_idx: int, threshold: float = 0.4) -> bool:
    """
    親指先端が特定のランドマークに近いかを
    手のサイズで正規化して判定する
    """
    def dist(a, b):
        return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    hand_size = dist(landmarks[0], landmarks[9]) + 1e-6
    return dist(landmarks[4], landmarks[target_idx]) / hand_size < threshold


class BaseGesture(ABC):
    name: str = ""
    label: str = ""

    @staticmethod
    @abstractmethod
    def detect(landmarks) -> bool:
        raise NotImplementedError