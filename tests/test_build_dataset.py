import unittest
from PIL import Image

from src.features.build_dataset import is_image_file, load_image


class TestBuildDataset(unittest.TestCase):
    def test_valid_image_ext(self):
        self.assertTrue(is_image_file('some.png'))
        self.assertTrue(is_image_file('some.jpg'))
        self.assertTrue(is_image_file('some.jpeg'))

    def test_invalid_image_ext(self):
        self.assertFalse(is_image_file('some.py'))
        self.assertFalse(is_image_file('some.cpp'))
        self.assertFalse(is_image_file('some'))

    def test_load_image(self):
        result = load_image('data/raw/BSDS300/images/train/2092.jpg')
        self.assertTrue(isinstance(result, Image.Image))
