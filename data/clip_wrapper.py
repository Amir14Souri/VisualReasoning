import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPExtractor:

    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16", use_safetensors=True
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def extract(self, images, texts):
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        text_out = self.model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        patch_tokens = vision_out.last_hidden_state[:, 1:, :]
        text_feature = text_out.pooler_output

        return patch_tokens, text_feature
