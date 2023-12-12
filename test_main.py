import pytest
from PIL import Image
from main import load_image_from_file, load_image_from_url, classify_image


def test_load_image_from_file():
    temp_file = "img/red.png"
    loaded_image = load_image_from_file(temp_file)
    assert isinstance(loaded_image, Image.Image)

def test_load_image_from_url():
    url = "https://avatars.githubusercontent.com/u/92524901?v=4"
    loaded_image = load_image_from_url(url)
    assert isinstance(loaded_image, Image.Image)

def test_classify_image():
    file_path = "img/red.png"
    image = Image.open(file_path)
    result = classify_image(image)
    assert isinstance(result, str)