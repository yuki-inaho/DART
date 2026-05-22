"""Shared base class for SAM3 multi-class predictors.

Extracts common utilities (empty results, mask IoU, NMS) that were
previously duplicated between Sam3MultiClassPredictor and
Sam3MultiClassPredictorFast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import PIL.Image
import torch
from jaxtyping import Bool, Float, Int
from torchvision.transforms import v2


class Sam3MultiClassPredictorBase(ABC):
    """Abstract base for multi-class SAM3 predictors.

    Provides shared infrastructure: image transforms, empty result
    construction, mask IoU, and greedy mask-based NMS.

    Subclasses implement `set_classes`, `set_image`, and `predict`.
    """

    def __init__(
        self,
        model,
        resolution: int = 1008,
        device: str = "cuda",
        detection_only: bool = False,
    ):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.detection_only = detection_only

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._class_names: list[str] | None = None
        self._num_classes: int = 0

    # ------------------------------------------------------------------
    # Abstract public API
    # ------------------------------------------------------------------

    @abstractmethod
    def set_classes(self, class_names: list[str]) -> None: ...

    @abstractmethod
    def set_image(
        self,
        image: PIL.Image.Image | torch.Tensor | np.ndarray,
        state: dict | None = None,
    ) -> dict: ...

    @abstractmethod
    def predict(
        self,
        state: dict,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
    ) -> dict: ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _empty_result(self, orig_h: int, orig_w: int) -> dict:
        """Return an empty predictions dict when nothing is detected."""
        if self.detection_only:
            return {
                "boxes": torch.zeros(0, 4, device=self.device),
                "masks": None,
                "masks_logits": None,
                "scores": torch.zeros(0, device=self.device),
                "class_ids": torch.zeros(0, device=self.device, dtype=torch.long),
                "class_names": [],
            }
        return {
            "boxes": torch.zeros(0, 4, device=self.device),
            "masks": torch.zeros(
                0, orig_h, orig_w, device=self.device, dtype=torch.bool
            ),
            "masks_logits": torch.zeros(0, 1, orig_h, orig_w, device=self.device),
            "scores": torch.zeros(0, device=self.device),
            "class_ids": torch.zeros(0, device=self.device, dtype=torch.long),
            "class_names": [],
        }

    @staticmethod
    def mask_iou(
        mask_a: Bool[torch.Tensor, "height width"],
        mask_b: Bool[torch.Tensor, "height width"],
    ) -> Float[torch.Tensor, ""]:
        """Compute IoU between two binary masks."""
        intersection = (mask_a & mask_b).sum().float()
        union = (mask_a | mask_b).sum().float()
        return intersection / union.clamp(min=1.0)

    def mask_nms(
        self,
        scores: Float[torch.Tensor, " K"],
        masks: Bool[torch.Tensor, "K height width"],
        class_ids: Int[torch.Tensor, " K"],
        iou_threshold: float,
        per_class: bool,
    ) -> Int[torch.Tensor, " kept"]:
        """Greedy mask-based NMS.

        Args:
            scores: (K,) detection scores.
            masks: (K, H, W) binary masks.
            class_ids: (K,) class assignments.
            iou_threshold: Suppress detections with IoU above this.
            per_class: If True, only suppress within same class.

        Returns:
            Indices of kept detections.
        """
        order = scores.argsort(descending=True)
        keep: list[int] = []

        for i in order.tolist():
            should_keep = True
            for j in keep:
                if per_class and class_ids[i] != class_ids[j]:
                    continue
                iou = self.mask_iou(masks[i], masks[j])
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(i)

        return torch.tensor(keep, device=scores.device, dtype=torch.long)
