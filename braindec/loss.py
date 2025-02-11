import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale):
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(logits_per_image), device=logits_per_image.device)

        return (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2
