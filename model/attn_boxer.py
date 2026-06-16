import math
import torch
import torch.nn as nn


class AttentionBoxer(nn.Module):

    def __init__(self, dim=256, coord_dim=2, vision_dim=768, text_dim=512):
        super().__init__()
        self.dim = dim
        self.query_proj = nn.Linear(text_dim, self.dim)
        self.key_proj = nn.Linear(vision_dim + coord_dim, self.dim)
        self.value_proj = nn.Linear(vision_dim + coord_dim, self.dim)

        patch_coords = [[c / 13.0, r / 13.0] for r in range(14) for c in range(14)]
        patch_coords = torch.tensor(patch_coords, dtype=torch.float32)
        self.register_buffer("coords", patch_coords)

    def forward(self, patch_tokens, question_feature):
        B, N, D = patch_tokens.shape
        coords = self.coords.unsqueeze(0).expand(B, -1, -1)
        patches = torch.cat([patch_tokens, coords], dim=-1)

        Q = self.query_proj(question_feature).unsqueeze(1)
        K = self.key_proj(patches)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)
        weights = torch.softmax(attn, dim=-1)

        return weights.squeeze(1)
