import unittest

from person_track import PersonTrack

OPERATOR = [500, 100, 700, 700]
BYSTANDER = [900, 150, 1100, 650]


class PersonTrackTest(unittest.TestCase):
    def test_holds_operator_when_a_more_confident_bystander_appears(self):
        track = PersonTrack()
        self.assertEqual(track.select([OPERATOR], 0), 0)
        # YOLO lists the bystander first (higher confidence).
        self.assertEqual(track.select([BYSTANDER, [510, 100, 710, 700]], .1), 1)

    def test_ambiguous_start_picks_nobody(self):
        track = PersonTrack()
        self.assertIsNone(track.select([BYSTANDER, OPERATOR], 0))
        self.assertEqual(track.select([OPERATOR], .1), 0)

    def test_lost_operator_is_not_replaced_within_hold(self):
        track = PersonTrack()
        track.select([OPERATOR], 0)
        self.assertIsNone(track.select([BYSTANDER], .5))
        self.assertIsNone(track.select([], .8))
        # After the hold a lone person is taken again.
        self.assertEqual(track.select([BYSTANDER], 1.6), 0)

    def test_empty_frame(self):
        self.assertIsNone(PersonTrack().select([], 0))


if __name__ == '__main__':
    unittest.main()
