import torch
import logging
from pathlib import Path

# Seed
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image parameters
IMG_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = 14

# Dataset sizes
TRAIN_SIZE = 500
VAL_SIZE = 100
TEST_SIZE = 200

# Image generation parameters
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR_RGB = {
    "red": (220, 40, 40),
    "blue": (40, 40, 220),
    "green": (40, 180, 40),
    "yellow": (230, 230, 40),
    "purple": (170, 60, 170),
    "orange": (240, 140, 40),
}
SHAPES = ["circle", "square", "triangle", "rectangle"]
POSITIONS = {
    "top-left": (37, 37),
    "top": (112, 37),
    "top-right": (187, 37),
    "left": (37, 112),
    "center": (112, 112),
    "right": (187, 112),
    "bottom-left": (37, 187),
    "bottom": (112, 187),
    "bottom-right": (187, 187),
}


# Logging setup
def setup_logging(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
