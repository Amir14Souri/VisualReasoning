import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from constants import DEVICE
from torch.utils.data import DataLoader

from model.attn_boxer import AttentionBoxer
from model.loss import TotalLoss
from training.trainer import Trainer
from data.dataset import SyntheticVQADataset
from data.utils import custom_collate


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


def main(args):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # wandb.init(project="attn-boxer", name="train_run)
    setup_logging(save_dir / "train.log")
    logging.info("Starting training...")

    model = AttentionBoxer().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = TotalLoss()

    train_set = SyntheticVQADataset("data/synthetic/train.json")
    val_set = SyntheticVQADataset("data/synthetic/val.json")
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_set, args.batch_size, shuffle=False, collate_fn=custom_collate
    )
    logging.info("Train and validation data loaders initialized.")

    trainer = Trainer(model, optimizer, criterion, DEVICE, args.num_epochs, save_dir)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="training/logs",
        help="Directory to save training results",
    )

    args = parser.parse_args()
    main(args)
