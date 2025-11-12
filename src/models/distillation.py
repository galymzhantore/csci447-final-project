from __future__ import annotations

import torch
from torch import nn


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.ce(student_logits, targets)
        if teacher_logits is not None:
            student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
            loss = self.alpha * self.kl(student_log_probs, teacher_probs) * (self.temperature**2) + (1 - self.alpha) * loss
        return loss


__all__ = ["DistillationLoss"]
