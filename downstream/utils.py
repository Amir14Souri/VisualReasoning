import string
from PIL import ImageDraw


def normalize_answer(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def draw_box(image, box):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="black", width=3)
    return image
