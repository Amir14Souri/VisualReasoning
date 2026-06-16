import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from training.utils import bbox_to_patch_mask
from data.clip_wrapper import CLIPExtractor


class Trainer:

    def __init__(
        self, model, optimizer, criterion, device, num_epochs=30, save_dir="train_run"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.clip = CLIPExtractor(self.device)

        self.best_val_loss = float("inf")
        self.best_model_epoch = -1
        self.train_loss_history = []
        self.val_loss_history = []

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            patches, q_feat = self.clip.extract(
                batch["images"], batch["target_phrases"]
            )
            heatmaps = self.model(patches, q_feat)
            target = bbox_to_patch_mask(batch["bboxes"]).to(self.device)

            loss = self.loss_fn(heatmaps, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate(self, epoch, val_loader):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}", leave=False):
                patches, q_feat = self.clip.extract(
                    batch["images"], batch["target_phrases"]
                )
                heatmaps = self.model(patches, q_feat)
                target = bbox_to_patch_mask(batch["bboxes"]).to(self.device)

                loss = self.loss_fn(heatmaps, target)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def fit(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.validate(epoch, val_loader)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            logging.info(
                f"Epoch {epoch+1:2d}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")

        self.model.load_state_dict(
            torch.load(self.save_dir / "best_model.pth", weights_only=True)
        )
        torch.save(self.model.state_dict(), "final_model.pth")

        logging.info(
            f"\nBest model from epoch {self.best_model_epoch} "
            f"with val loss={self.best_val_loss:.5f}"
        )
        self.plot_losses()

    def plot_losses(self):
        plt.plot(self.train_loss_history, label="Train Loss", color="b")
        plt.plot(self.val_loss_history, label="Val Loss", color="g")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / "loss_curves.png")
