import torch
import random
from PIL import ImageDraw
from constants import IMG_SIZE, PATCH_SIZE, SEED


def heatmap_to_bbox(heatmap, k=9):
    h = w = int(IMG_SIZE / PATCH_SIZE)
    flat = heatmap.flatten()
    order = torch.argsort(flat, descending=True)
    top_k_indices = order[:k]

    selected = torch.zeros(h, w, dtype=torch.bool)
    selected[top_k_indices[0] // w, top_k_indices[0] % w] = True
    count = 1

    while count < k:
        added = False
        for idx in top_k_indices:
            r, c = idx // w, idx % w
            if count >= k:
                break
            if selected[r, c]:
                continue
            neighbors = selected[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
            if neighbors.any():
                selected[r, c] = True
                count += 1
                added = True
        if not added:
            break

    rows, cols = torch.where(selected)
    y1, y2 = rows.min() * 16, (rows.max() + 1) * 16
    x1, x2 = cols.min() * 16, (cols.max() + 1) * 16

    return torch.tensor([x1, y1, x2, y2])


def eval_draw(image, pred_bbox, gt_bbox, save_path):
    image = image.convert("RGB").resize((224, 224))
    draw = ImageDraw.Draw(image)

    # Draw 14x14 grid (thin black lines)
    for i in range(15):
        draw.line([(i * PATCH_SIZE, 0), (i * PATCH_SIZE, 224)], fill="black", width=1)
        draw.line([(0, i * PATCH_SIZE), (224, i * PATCH_SIZE)], fill="black", width=1)

    # Draw predicted bbox (blue)
    draw.rectangle(
        [pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]],
        outline="blue",
        width=2,
    )

    # Draw ground truth bbox (green)
    draw.rectangle(
        [gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]], outline="green", width=2
    )

    image.save(save_path)


def compute_iou(a, b):
    # top-left of intersection
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    # bottom-right of intersection
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = area_a + area_b - intersection
    return intersection / (union + 1e-6)


def center_in_target(pred, gt):
    cx = (pred[0] + pred[2]) / 2
    cy = (pred[1] + pred[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def random_box():
    random.seed(SEED)
    w = random.randint(32, 64)
    h = random.randint(32, 64)
    x = random.randint(0, 224 - w)
    y = random.randint(0, 224 - h)
    return [x, y, x + w, y + h]


def center_box():
    return [72, 72, 152, 152]
