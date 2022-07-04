import unittest

from app.main import CURR_TRACK, FPS, HEIGHT, TRACK_ID, WIDTH


class MainTest(unittest.TestCase):
    def test_track_id(self):
        """
        test collision

        """

        if TRACK_ID == 0:
            self.assertEquals(CURR_TRACK, "AutonomoHermanosRodriguez")

        elif TRACK_ID == 1:
            self.assertEquals(CURR_TRACK, "CircuitOfTheAmericas")

        elif TRACK_ID == 2:
            self.assertEquals(CURR_TRACK, "Monaco")

        elif TRACK_ID == 3:
            self.assertEquals(CURR_TRACK, "Monza")

        else:
            assert False

    def test_fps(self):
        self.assertEquals(FPS, 28)
