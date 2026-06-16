import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from constants import DEVICE, save_json
from data.clip_wrapper import CLIPExtractor
from downstream.utils import normalize_answer, draw_box
from downstream.qwen_client import QwenClient
from evaluation.utils import heatmap_to_bbox


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
            for batch in tqdm(test_loader, desc="Downstream VQA"):
                images = batch["images"]
                questions = batch["questions"]
                gt_answers = [normalize_answer(a) for a in batch["answers"]]
                families = batch["families"]

                self._evaluate_full(questions, images, gt_answers, families)
                self._evaluate_attn(
                    questions,
                    images,
                    batch["target_phrases"],
                    gt_answers,
                    families,
                    start,
                )
                self._evaluate_oracle(
                    questions, images, batch["bboxes"], gt_answers, families, start
                )
                start += len(batch)

        save_json(self.results, self.save_dir / "detailed_results.json")
        return self.results

    def _evaluate_full(self, questions, images, gt_answers, families):
        prompts = [
            (
                "Answer the question using only visible evidence.\n"
                "Answer with only one word, one letter, or one number.\n"
                f"Question: {q}"
            )
            for q in questions
        ]

        preds = self.qwen.ask(images, prompts)
        self._store("full", preds, gt_answers, families)

    def _evaluate_attn(
        self, questions, images, target_phrases, gt_answers, families, start_idx
    ):
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
        self._store("attention", preds, gt_answers, families)

    def _evaluate_oracle(
        self, questions, images, bboxes, gt_answers, families, start_idx
    ):
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
        self._store("oracle", preds, gt_answers, families)

    def _store(self, setting, preds, gt_answers, families):
        for pred, gt, family in zip(preds, gt_answers, families):
            pred = normalize_answer(pred)
            self.results[setting].append(
                {"pred": pred, "gt": gt, "correct": pred == gt, "family": family}
            )

    def print_results(self):
        settings = ["family/setting", "full", "attention", "oracle"]
        logging.info(
            f"You can see overall results in {self.save_dir / 'overall_results.json'} too."
        )

        logging.info("+" + ("-" * 16 + "+") * 4)
        logging.info("|" + "|".join([f" {s.upper():^14} " for s in settings]) + "|")
        logging.info("+" + ("-" * 16 + "+") * 4)

        accuracies = {
            s: np.mean([r["correct"] for r in self.results[s]]) * 100
            for s in settings[1:]
        }
        save_json(accuracies, self.save_dir / "overall_results.json")
        self._print_row("overall", accuracies, settings)

        for family in ["attribute", "text"]:
            subset = {
                s: [r for r in self.results[s] if r["family"] == family]
                for s in settings[1:]
            }
            accuracies = {
                s: np.mean([r["correct"] for r in subset[s]]) * 100
                for s in settings[1:]
            }
            self._print_row(family, accuracies, settings)

    def _print_row(self, row, accuracies, settings):
        row = row.upper()
        logging.info(
            "|"
            + (f" {row:^14} " + "|")
            + "|".join(
                [(f"{str(round(accuracies[s], 1))}%").center(16) for s in settings[1:]]
            )
            + "|"
        )
        logging.info("+" + ("-" * 16 + "+") * 4)
