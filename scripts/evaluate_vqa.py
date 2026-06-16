import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from constants import DEVICE, setup_logging
from model.attn_boxer import AttentionBoxer
from data.dataset import SyntheticVQADataset
from data.utils import custom_collate
from downstream.vqa_eval import DownstreamVQAEvaluator


def main(args):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(save_dir / "eval.log")
    logging.info("Starting evaluation on test data for downstream task...")

    model = AttentionBoxer().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    test_set = SyntheticVQADataset("data/synthetic/test.json")
    test_loader = DataLoader(
        test_set, args.batch_size, shuffle=False, collate_fn=custom_collate
    )
    logging.info("Test data loader initialized.")

    evaluator = DownstreamVQAEvaluator(model, k=args.k, save_dir=save_dir)
    evaluator.run(test_loader)
    evaluator.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for test",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Value of k for heatmap to bbox conversion",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="downstream/vqa_results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()
    main(args)
