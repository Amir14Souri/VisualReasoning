import os
import torch
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from constants import save_json
from data.clip_wrapper import CLIPExtractor
from evaluation.utils import heatmap_to_bbox, compute_iou, center_in_target, eval_draw


class KTuner:

    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.clip = CLIPExtractor(self.device)

    def collect(self, val_loader):
        self.model.eval()
        all_data = []

        with torch.no_grad():
            for sample in tqdm(val_loader, desc="Collecting"):
                patches, q_feat = self.clip.extract(
                    sample["images"], sample["target_phrases"]
                )
                heatmaps = self.model(patches, q_feat)

                for i in range(len(heatmaps)):
                    all_data.append(
                        {
                            "heatmap": heatmaps[i].cpu(),
                            "bbox": sample["bboxes"][i],
                            "image": sample["images"][i],
                        }
                    )

        return all_data

    def evaluate_k(self, all_data, k_values):
        best_k, best_iou = None, 0
        results = defaultdict(list)

        for k in k_values:
            ious = []
            iou_at30_count = 0
            iou_at50_count = 0
            cit_count = 0
            for data in all_data:
                pred_bbox = heatmap_to_bbox(data["heatmap"], k)
                iou = compute_iou(pred_bbox, data["bbox"])
                cit = center_in_target(pred_bbox, data["bbox"])
                ious.append(iou)
                iou_at30_count += int(iou >= 0.3)
                iou_at50_count += int(iou >= 0.5)
                cit_count += int(cit)

            avg_iou = sum(ious) / len(ious)
            iou_at30 = iou_at30_count / len(ious)
            iou_at50 = iou_at50_count / len(ious)
            avg_cit = cit_count / len(ious)
            results[f"k={k}"] = {
                "avg_iou": float(avg_iou),
                "iou_at30": float(iou_at30),
                "iou_at50": float(iou_at50),
                "avg_cit": float(avg_cit),
            }
            logging.info(
                f"k={k:2d}: Avg IoU = {avg_iou:.4f}, Avg CiT = {avg_cit:.2f}, IoU@0.3 = {iou_at30:.2f}, IoU@0.5 = {iou_at50:.2f}"
            )

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_k = k

        save_json(results, self.save_dir / "k_tuning_results.json")
        logging.info(f"Best k={best_k} | IoU={best_iou:.4f}")
        return best_k, best_iou

    def visualize(self, all_data, best_k):
        for i, d in enumerate(tqdm(all_data, desc="Saving images")):
            pred = heatmap_to_bbox(d["heatmap"], best_k)
            eval_draw(d["image"], pred, d["bbox"], f"{self.save_dir}/{i}.png")
