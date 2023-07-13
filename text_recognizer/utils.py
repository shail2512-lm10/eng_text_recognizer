from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
import smart_open


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype="uint8")[y]

def read_img(img_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(img_uri, 'rb') as img_file:
        return read_img_file(img_file, grayscale)
    
def read_img_file(img_file, grayscale=False) -> Image:
    with Image.open(img_file) as img:
        if grayscale:
            img = img.convert(mode="L")
        else:
            img = img.convert(mode=img.mode)
        return img


