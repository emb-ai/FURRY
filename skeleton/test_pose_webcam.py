import unittest
from types import SimpleNamespace

from pose_webcam import joints_from_landmarks, path_length, summarize


def mark(x, y, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility, presence=visibility)


class TrajectoryTest(unittest.TestCase):
    def test_pixels_come_from_normalized_landmarks(self):
        landmarks = [mark(0, 0, 0)] * 33
        landmarks[0] = mark(0.5, 0.25, 0.8)
        joints = joints_from_landmarks(landmarks, 1280, 720)
        self.assertEqual(joints["nose"]["uv"], [640.0, 180.0])
        self.assertEqual(joints["nose"]["visibility"], 0.8)
        self.assertEqual(len(joints), 33)

    def test_leg_path_ignores_low_visibility(self):
        hidden = {"left_ankle": {"uv": [0, 0], "visibility": 0.1}}
        a = {"joints": {"left_ankle": {"uv": [0, 0], "visibility": 1}}}
        b = {"joints": {"left_ankle": {"uv": [30, 40], "visibility": 1}}}
        c = {"joints": hidden["left_ankle"]}
        self.assertEqual(path_length([a, b], "left_ankle"), 50.0)
        self.assertEqual(path_length([a, c], "left_ankle"), 0.0)

    def test_summary_counts_visible_legs(self):
        def frame(vis):
            joints = {
                name: {"uv": [1, 1], "visibility": vis}
                for name in (
                    "left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle", "left_heel", "right_heel",
                    "left_foot_index", "right_foot_index", "nose",
                )
            }
            return {"joints": joints}

        summary = summarize([frame(0.9), frame(0.1)], 640, 480)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["units"], "pixels")
        self.assertEqual(summary["legs"]["left_knee"]["visible_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
