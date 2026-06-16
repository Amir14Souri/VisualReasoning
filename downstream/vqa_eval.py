from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from constants import DEVICE, save_json
from data.clip_wrapper import CLIPExtractor
from downstream.utils import normalize_answer, draw_box
from evaluation.utils import heatmap_to_bbox
from downstream.qwen_client import QwenClient


class DownstreamVQAEvaluator:
    def __init__(self, model, k, save_dir):
        self.model = model
        self.k = k
        self.save_dir = save_dir

        self.clip = CLIPExtractor(DEVICE)
        self.qwen = QwenClient()
        self.results = defaultdict(list)

    def run(self, test_loader):
        self.model.eval()
        start = 0

        with torch.no_grad():
            for sample in tqdm(test_loader, desc="Downstream VQA"):
                images = sample["images"]
                questions = sample["questions"]
                gt_answers = [normalize_answer(a) for a in sample["answers"]]

                self._evaluate_full(questions, images, gt_answers)
                self._evaluate_attn(
                    questions, images, sample["target_phrases"], gt_answers, start
                )
                self._evaluate_oracle(
                    questions, images, sample["bboxes"], gt_answers, start
                )
                start += len(sample)

        save_json(self.results, self.save_dir / "detailed_results.json")
        return self.results

    def _evaluate_full(self, questions, images, gt_answers):
        prompts = [
            (
                "Answer the question using only visible evidence.\n"
                "Answer with only one word, one letter, or one number.\n"
                f"Question: {q}"
            )
            for q in questions
        ]

        preds = self.qwen.ask(images, prompts)
        self._store("full", preds, gt_answers)

    def _evaluate_attn(self, questions, images, target_phrases, gt_answers, start_idx):
        patches, q_feat = self.clip.extract(images, target_phrases)
        heatmaps = self.model(patches, q_feat)

        boxed_images = []
        for idx, (image, heatmap) in enumerate(zip(images, heatmaps)):
            boxed_image = draw_box(image, heatmap_to_bbox(heatmap.cpu(), self.k))
            boxed_images.append(boxed_image)
            img = boxed_image.convert("RGB").resize((224, 224))
            img.save(self.save_dir / f"{start_idx + idx}_attn.png")

        prompts = [
            (
                "The image contains a highlighted box that may contain the relevant evidence.\n"
                "Ignore the color of the border. The border only marks a region.\n"
                "Use the highlighted region if it is relevant.\n\n"
                "Answer with only one word, one letter, or one number.\n"
                f"Question: {q}"
            )
            for q in questions
        ]

        preds = self.qwen.ask(boxed_images, prompts)
        self._store("attention", preds, gt_answers)

    def _evaluate_oracle(self, questions, images, bboxes, gt_answers, start_idx):
        oracle_images = []
        for idx, (image, bbox) in enumerate(zip(images, bboxes)):
            oracle_image = draw_box(image, bbox)
            oracle_images.append(oracle_image)
            img = oracle_image.convert("RGB").resize((224, 224))
            img.save(self.save_dir / f"{start_idx + idx}_oracle.png")

        prompts = [
            (
                "The image contains a highlighted box that may contain the relevant evidence.\n"
                "Ignore the color of the border. The border only marks a region.\n"
                "Use the highlighted region if it is relevant.\n\n"
                "Answer with only one word, one letter, or one number.\n"
                f"Question: {q}"
            )
            for q in questions
        ]

        preds = self.qwen.ask(oracle_images, prompts)
        self._store("oracle", preds, gt_answers)

    def _store(self, setting, preds, gt_answers):
        for pred, gt in zip(preds, gt_answers):
            pred = normalize_answer(pred)
            self.results[setting].append(
                {"pred": pred, "gt": gt, "correct": pred == gt}
            )

    def print_results(self):
        print("+" + "-" * 20 + "+")
        print("|" + "QWEN VQA RESULTS".center(20) + "|")
        print("+" + "-" * 20 + "+")
        results = {}

        for setting in ["full", "attention", "oracle"]:
            accuracy = np.mean([r["correct"] for r in self.results[setting]]) * 100
            results[setting] = accuracy
            print(f"| {setting.upper():<13}{str(round(accuracy, 1)):<4}% |")
        print("+" + "-" * 20 + "+")

        save_json(results, self.save_dir / "overall_results.json")
