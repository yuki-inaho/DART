"""Regression tests for predictor base class and shared utilities.

These tests do NOT require a model checkpoint — they verify the
refactored shared code (mask_iou, mask_nms, _empty_result, inheritance)
works identically to the original duplicated implementations.
"""

from typing import Union

import numpy as np
import PIL.Image
import torch
import pytest
from beartype import beartype

from sam3.model.predictor_base import Sam3MultiClassPredictorBase
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class ConcretePredictor(Sam3MultiClassPredictorBase):
    """Minimal concrete subclass for testing base methods."""

    @beartype
    def set_classes(self, class_names: list[str]) -> None:
        self._class_names = class_names
        self._num_classes = len(class_names)

    @beartype
    def set_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        state: dict | None = None,
    ) -> dict:
        return {}

    @beartype
    def predict(self, state: dict, confidence_threshold: float = 0.3, nms_threshold: float = 0.7) -> dict:
        return self._empty_result(100, 100)


@pytest.fixture
def predictor():
    return ConcretePredictor(model=None, resolution=504, device="cpu", detection_only=False)


@pytest.fixture
def predictor_det_only():
    return ConcretePredictor(model=None, resolution=504, device="cpu", detection_only=True)


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_sequential_inherits_base(self):
        assert issubclass(Sam3MultiClassPredictor, Sam3MultiClassPredictorBase)

    def test_fast_inherits_base(self):
        assert issubclass(Sam3MultiClassPredictorFast, Sam3MultiClassPredictorBase)

    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            Sam3MultiClassPredictorBase(model=None)


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------

class TestMaskIoU:
    def test_identical_masks(self):
        mask = torch.ones(10, 10, dtype=torch.bool)
        assert Sam3MultiClassPredictorBase.mask_iou(mask, mask).item() == pytest.approx(1.0)

    def test_no_overlap(self):
        a = torch.zeros(10, 10, dtype=torch.bool)
        b = torch.zeros(10, 10, dtype=torch.bool)
        a[:5] = True
        b[5:] = True
        assert Sam3MultiClassPredictorBase.mask_iou(a, b).item() == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = torch.zeros(10, 10, dtype=torch.bool)
        b = torch.zeros(10, 10, dtype=torch.bool)
        a[:7] = True   # 70 pixels
        b[3:] = True   # 70 pixels
        # intersection: rows 3-6 = 40, union: rows 0-9 = 100
        iou = Sam3MultiClassPredictorBase.mask_iou(a, b).item()
        assert iou == pytest.approx(40.0 / 100.0)

    def test_empty_masks(self):
        a = torch.zeros(10, 10, dtype=torch.bool)
        b = torch.zeros(10, 10, dtype=torch.bool)
        # union is 0, clamp(min=1) prevents div-by-zero → result 0
        assert Sam3MultiClassPredictorBase.mask_iou(a, b).item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mask_nms
# ---------------------------------------------------------------------------

class TestMaskNMS:
    def test_no_suppression_different_classes(self, predictor):
        scores = torch.tensor([0.9, 0.8])
        masks = torch.ones(2, 10, 10, dtype=torch.bool)  # identical masks
        class_ids = torch.tensor([0, 1])  # different classes
        keep = predictor.mask_nms(scores, masks, class_ids, iou_threshold=0.5, per_class=True)
        assert len(keep) == 2

    def test_suppression_same_class(self, predictor):
        scores = torch.tensor([0.9, 0.8])
        masks = torch.ones(2, 10, 10, dtype=torch.bool)  # identical → IoU=1
        class_ids = torch.tensor([0, 0])  # same class
        keep = predictor.mask_nms(scores, masks, class_ids, iou_threshold=0.5, per_class=True)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept

    def test_cross_class_suppression(self, predictor):
        scores = torch.tensor([0.9, 0.8])
        masks = torch.ones(2, 10, 10, dtype=torch.bool)
        class_ids = torch.tensor([0, 1])
        keep = predictor.mask_nms(scores, masks, class_ids, iou_threshold=0.5, per_class=False)
        assert len(keep) == 1

    def test_empty_input(self, predictor):
        scores = torch.zeros(0)
        masks = torch.zeros(0, 10, 10, dtype=torch.bool)
        class_ids = torch.zeros(0, dtype=torch.long)
        keep = predictor.mask_nms(scores, masks, class_ids, iou_threshold=0.5, per_class=True)
        assert len(keep) == 0

    def test_order_by_score(self, predictor):
        scores = torch.tensor([0.3, 0.9, 0.6])
        m = torch.zeros(3, 10, 10, dtype=torch.bool)
        m[0, :3] = True
        m[1, 3:6] = True
        m[2, 6:9] = True  # no overlap between any pair
        class_ids = torch.tensor([0, 0, 0])
        keep = predictor.mask_nms(scores, m, class_ids, iou_threshold=0.5, per_class=True)
        assert len(keep) == 3
        assert keep[0] == 1  # highest score first


# ---------------------------------------------------------------------------
# _empty_result
# ---------------------------------------------------------------------------

class TestEmptyResult:
    def test_detection_only(self, predictor_det_only):
        r = predictor_det_only._empty_result(480, 640)
        assert r["boxes"].shape == (0, 4)
        assert r["masks"] is None
        assert r["masks_logits"] is None
        assert r["scores"].shape == (0,)
        assert r["class_ids"].shape == (0,)
        assert r["class_names"] == []

    def test_with_masks(self, predictor):
        r = predictor._empty_result(480, 640)
        assert r["boxes"].shape == (0, 4)
        assert r["masks"].shape == (0, 480, 640)
        assert r["masks_logits"].shape == (0, 1, 480, 640)
        assert r["scores"].shape == (0,)

    def test_device_consistency(self, predictor):
        r = predictor._empty_result(100, 100)
        assert r["boxes"].device.type == "cpu"
        assert r["scores"].device.type == "cpu"
        assert r["masks"].device.type == "cpu"


# ---------------------------------------------------------------------------
# beartype runtime checks
# ---------------------------------------------------------------------------

class TestBeartype:
    def test_set_classes_rejects_non_list(self, predictor):
        with pytest.raises(Exception):
            predictor.set_classes("not a list")

    def test_set_classes_rejects_non_str_items(self, predictor):
        with pytest.raises(Exception):
            predictor.set_classes([1, 2, 3])

    def test_set_classes_accepts_valid(self, predictor):
        predictor.set_classes(["apple", "banana"])
        assert predictor._num_classes == 2

    def test_set_image_rejects_invalid_type(self, predictor):
        with pytest.raises(Exception):
            predictor.set_image("not an image")


# ---------------------------------------------------------------------------
# loading module
# ---------------------------------------------------------------------------

class TestLoading:
    def test_import(self):
        from sam3.loading import load_model
        assert callable(load_model)

    def test_invalid_imgsz(self):
        from sam3.loading import load_model
        with pytest.raises(Exception):
            load_model(device="cpu", imgsz=100, checkpoint_path="nonexistent.pt")
