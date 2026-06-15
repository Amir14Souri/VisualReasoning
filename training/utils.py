import torch


def bbox_to_patch_mask(bboxes):
    N = bboxes.shape[0]
    device = bboxes.device

    cols = torch.arange(14, device=device).float()
    rows = torch.arange(14, device=device).float()

    px1, py1 = cols * 16, rows * 16
    px2, py2 = px1 + 16, py1 + 16

    x1 = bboxes[:, 0].view(N, 1, 1)
    y1 = bboxes[:, 1].view(N, 1, 1)
    x2 = bboxes[:, 2].view(N, 1, 1)
    y2 = bboxes[:, 3].view(N, 1, 1)

    px1 = px1.view(1, 1, 14)
    py1 = py1.view(1, 14, 1)
    px2 = px2.view(1, 1, 14)
    py2 = py2.view(1, 14, 1)

    overlap_x = (px1 < x2) & (x1 < px2)  # [N, 14, 14]
    overlap_y = (py1 < y2) & (y1 < py2)  # [N, 14, 14]
    overlap = overlap_x & overlap_y  # [N, 14, 14]

    mask = overlap.float().view(N, -1)
    return mask
