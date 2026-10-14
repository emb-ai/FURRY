"""Keep the same person across frames when several are in view.

YOLO orders detections by confidence, so taking the first one lets a bystander
replace the operator between frames (ep3, 2026-10-12: a closer second person
drove the camera legs for ~18 s). The track holds the operator by box centre;
when it cannot tell who is who it returns no person instead of guessing, and
the headset keeps Meta legs for those frames.
"""
import numpy as np

GATE = .5     # max centre jump between frames, in heights of the last box
HOLD_S = 1.   # a track older than this is dropped; a lone person is then taken


def centre(boxes):
    b = np.asarray(boxes, float)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


class PersonTrack:
    def __init__(self, gate=GATE, hold_s=HOLD_S):
        self.gate, self.hold_s = gate, hold_s
        self.box, self.seen = None, -np.inf

    def select(self, boxes, now):
        """Index into boxes ([N, 4] xyxy px) of the tracked person, or None."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return None
        if self.box is not None and now - self.seen <= self.hold_s:
            jump = np.linalg.norm(centre(boxes) - centre(self.box), axis=1)
            pick = int(np.argmin(jump))
            if jump[pick] > self.gate * (self.box[3] - self.box[1]):
                return None
        elif len(boxes) == 1:
            pick = 0
        else:
            return None
        self.box, self.seen = boxes[pick], now
        return pick
