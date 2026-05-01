from .base import BaseGesture, get_finger_states, thumb_near

class PointingGesture(BaseGesture):
    name = "pointing"
    label = "Pointing"

    @staticmethod
    def detect(landmarks) -> bool:
        f = get_finger_states(landmarks)
        index_only = f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]
        thumb_tucked = thumb_near(landmarks, target_idx=9, threshold=0.5)
        return index_only and thumb_tucked