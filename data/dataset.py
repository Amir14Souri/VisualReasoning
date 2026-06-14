import json
from torch.utils.data import Dataset
from PIL import Image


class SyntheticVQADataset(Dataset):

    def __init__(self, json_file):
        self.data = json.load(open(json_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        return {
            "image": image,
            "question": item["question"],
            "target_phrase": item["target_phrase"],
            "bbox": torch.tensor(item["target_bbox"], dtype=torch.float32),
            "answer": item["answer"],
            "family": item["family"],
        }
