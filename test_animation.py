import unittest
import logging

from plot_weights import ani_frame

logging.basicConfig(level=logging.INFO)
class MyTestCase(unittest.TestCase):
    def test_animation(self):
        ani_frame()


if __name__ == '__main__':
    unittest.main()
