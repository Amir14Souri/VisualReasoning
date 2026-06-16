import pandas as pd
from collections import defaultdict
import torch
import logging
import json
from tqdm import tqdm
from data.clip_wrapper import CLIPExtractor
from evaluation.utils import (
    heatmap_to_bbox,
    compute_iou,
    center_in_target,
    random_box,
    center_box,
)
from constants import save_json


class TestEvaluator:

    def __init__(self, model, device, k, save_dir):
        self.model = model
        self.device = device
        self.k = k
        self.clip = CLIPExtractor(self.device)
        self.save_dir = save_dir

    def run(self, test_loader):
        logging.info(f"Evaluating using k={self.k}")
        self.model.eval()
        results = defaultdict(list)

        with torch.no_grad():
            for sample in tqdm(test_loader, desc="Testing"):
                images = sample["images"]
                gt_boxes = sample["bboxes"]
                families = sample["families"]

                patches, qfeat = self.clip.extract(images, sample["target_phrases"])

                heatmaps = self.model(patches, qfeat)

                for i in range(len(images)):

                    heatmap = heatmaps[i].cpu()
                    gt_box = gt_boxes[i]
                    family = families[i]

                    # --- Attention model ---
                    pred_box = heatmap_to_bbox(heatmap, self.k)
                    results["attention"].append(self.eval(pred_box, gt_box, family))

                    # --- Random baseline ---
                    rand_box = random_box()
                    results["random"].append(self.eval(rand_box, gt_box, family))

                    # --- Center baseline ---
                    cbox = center_box()
                    results["center"].append(self.eval(cbox, gt_box, family))

        save_json(results, self.save_dir / "detailed_results.json")
        return results

    def eval(self, pred_box, gt_box, family):
        iou = compute_iou(pred_box, gt_box)
        cit = center_in_target(pred_box, gt_box)

        return {
            "iou": float(iou),
            "iou03": int(iou >= 0.3),
            "iou05": int(iou >= 0.5),
            "cit": int(cit),
            "family": family,
        }

    @staticmethod
    def summarize(records):
        df = pd.DataFrame(records)
        return {
            "Mean IoU": df["iou"].mean(),
            "IoU@0.3": df["iou03"].mean(),
            "IoU@0.5": df["iou05"].mean(),
            "Center-in-Target": df["cit"].mean(),
        }

    def print_results(self, results):
        methods = ["family/method", "attention", "random", "center"]
        overall_results = {m: self.summarize(results[m]) for m in methods[1:]}
        save_json(overall_results, self.save_dir / "overall_results.json")
        logging.info(
            f"You can see overall results in {self.save_dir / 'overall_results.json'} too."
        )

        logging.info("+" + ("-" * 24 + "+") * 4)
        logging.info("|" + "|".join([f" {m.upper():^22} " for m in methods]) + "|")
        logging.info("+" + ("-" * 24 + "+") * 4)

        self._print_row("overall", overall_results, methods)

        for family in ["attribute", "text"]:
            subset = {
                m: [r for r in results[m] if r["family"] == family] for m in methods[1:]
            }
            stats = {m: self.summarize(subset[m]) for m in methods[1:]}
            self._print_row(family, stats, methods)

    def _print_row(self, row, stats, methods):
        logging.info("|" + (" " * 24 + "|") * 4)
        for i, metric in enumerate(stats[methods[1]].keys()):
            row = row.upper() if i == 0 else " " * 22
            logging.info(
                "|"
                + (f" {row:^22} " + "|")
                + "|".join(
                    [
                        f" {metric:<17}{str(round(stats[m][metric], 3)):<5} "
                        for m in methods[1:]
                    ]
                )
                + "|"
            )
        logging.info("|" + (" " * 24 + "|") * 4)
        logging.info("+" + ("-" * 24 + "+") * 4)
