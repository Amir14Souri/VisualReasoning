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
from evaluation.tuner import KTuner


def main(args):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(save_dir / "eval.log")
    logging.info("Starting evaluation to tune value of k...")

    model = AttentionBoxer().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    val_set = SyntheticVQADataset("data/synthetic/val.json")
    val_loader = DataLoader(
        val_set, args.batch_size, shuffle=False, collate_fn=custom_collate
    )
    logging.info("Validation data loader initialized.")

    tuner = KTuner(model, DEVICE, save_dir)
    all_data = tuner.collect(val_loader)
    best_k, best_iou = tuner.evaluate_k(all_data, [1, 4, 9, 16, 25, 36])
    tuner.visualize(all_data, best_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation/val_results",
        help="Directory to save validation results",
    )

    args = parser.parse_args()
    main(args)
