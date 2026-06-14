import json
import math
import random
import string

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from constants import *


class DataGenerator:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        random.seed(SEED)

    def draw_shape(self, draw, shape, bbox, color):
        x1, y1, x2, y2 = bbox

        if shape == "circle":
            draw.ellipse(bbox, fill=color)
        elif shape in ["square", "rectangle"]:
            draw.rectangle(bbox, fill=color)
        elif shape == "triangle":
            points = [((x1 + x2) // 2, y1), (x1, y2), (x2, y2)]
            draw.polygon(points, fill=color)

    def create_object(self, position_name):
        cx, cy = POSITIONS[position_name]
        cx += random.randint(-10, 10)
        cy += random.randint(-10, 10)
        size = random.randint(24, 48)
        bbox = [cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2]

        obj = {
            "shape": random.choice(SHAPES),
            "color": random.choice(COLORS),
            "char": random.choice(
                list(string.ascii_uppercase[:10]) + list("0123456789")
            ),
            "position": position_name,
            "bbox": bbox,
        }
        return obj

    def add_distractors(self, draw):
        for _ in range(random.randint(5, 10)):
            x1 = random.randint(0, IMG_SIZE)
            y1 = random.randint(0, IMG_SIZE)
            length = random.randint(5, 25)
            angle = random.uniform(0, 2 * math.pi)
            x2 = x1 + int(length * math.cos(angle))
            y2 = y1 + int(length * math.sin(angle))
            color = random.choice(list(COLOR_RGB.values()))
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 3))

        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, IMG_SIZE - 20)
            y1 = random.randint(0, IMG_SIZE - 20)
            x2 = x1 + random.randint(10, 25)
            y2 = y1 + random.randint(10, 25)
            color = random.choice(list(COLOR_RGB.values()))
            draw.ellipse([x1, y1, x2, y2], outline=color, width=random.randint(1, 2))

    def create_questions(self, target):
        family = random.choice(["attribute", "text"])

        if family == "attribute":
            q_type = random.choice(["color", "shape"])

            if q_type == "color":
                question = (
                    f"What color is the {target['shape']} in the {target['position']}?"
                )
                answer = target["color"]
                target_phrase = f"{target['shape']} in the {target['position']}"
            else:
                question = f"What shape is the {target['color']} object in the {target['position']}?"
                answer = target["shape"]
                target_phrase = f"{target['color']} object in the {target['position']}"
        else:
            char_type = "digit" if target["char"].isdigit() else "letter"
            question = f"What {char_type} is inside the {target['color']} {target['shape']} in the {target['position']}?"
            answer = target["char"]
            target_phrase = (
                f"{target['color']} {target['shape']} in the {target['position']}"
            )

        return question, answer, target_phrase, family

    def generate_example(self, idx, split):
        # Create image and objects
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
        draw = ImageDraw.Draw(img)

        objects = []
        for pos in POSITIONS:
            obj = self.create_object(pos)
            objects.append(obj)

        # Draw objects and characters
        font = ImageFont.load_default(18)

        for obj in objects:
            self.draw_shape(draw, obj["shape"], obj["bbox"], COLOR_RGB[obj["color"]])

            x1, y1, x2, y2 = obj["bbox"]
            draw.text(
                (
                    (x1 + x2) // 2 - 8 + random.randint(-2, 2),
                    (y1 + y2) // 2 - 12 + random.randint(-2, 2),
                ),
                obj["char"],
                fill="black",
                font=font,
            )

        # Add random distractor stuff
        self.add_distractors(draw)

        # Create questions
        target = random.choice(objects)
        question, answer, target_phrase, family = self.create_questions(target)

        # Save image
        image_path = self.data_dir / split / f"{idx}.png"
        image_path.parent.mkdir(exist_ok=True, parents=True)
        img.save(image_path)

        return {
            "image_path": str(image_path),
            "question": question,
            "target_phrase": target_phrase,
            "answer": answer,
            "target_bbox": target["bbox"],
            "family": family,
        }

    def generate_split(self, split, n):
        records = []
        for i in tqdm(range(n), f"{split} set"):
            records.append(self.generate_example(i, split))

        with open(self.data_dir / f"{split}.json", "w") as f:
            json.dump(records, f, indent=2)

        return records
