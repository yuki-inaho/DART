"""Unified model loading for DART demos and scripts.

Consolidates the model loading logic previously duplicated between
demo_multiclass.py and demo_video.py into a single function.
"""

from __future__ import annotations

from beartype import beartype


@beartype
def load_model(
    *,
    device: str = "cuda",
    checkpoint_path: str | None = None,
    efficient_backbone: str | None = None,
    efficient_model: str | None = None,
    skip_blocks: set[int] | None = None,
    mask_blocks: list[str] | None = None,
    text_cache_path: str | None = None,
    trt_engine_path: str | None = None,
    trt_enc_dec_engine_path: str | None = None,
    detection_only: bool = False,
    imgsz: int = 1008,
):
    """Load a SAM3 model with the appropriate configuration.

    Handles all model variants: full ViT-H, pruned, EfficientSAM3,
    and TRT-only stub.

    Returns:
        The loaded model (on the specified device, in eval mode).
    """
    import os

    from sam3.efficient_backbone import build_efficientsam3_model
    from sam3.model_builder import (
        build_pruned_sam3_image_model,
        build_sam3_image_model,
        load_pruned_config,
    )

    # TRT-only mode: no checkpoint needed when text cache exists
    text_cache_exists = text_cache_path and os.path.exists(text_cache_path)
    use_trt_only = (
        text_cache_exists
        and trt_engine_path
        and trt_enc_dec_engine_path
        and detection_only
        and checkpoint_path is None
    )

    if use_trt_only:
        from sam3.model.sam3_multiclass_fast import _TRTModelStub

        print(f"Using TRT-only mode on {device} (no checkpoint — text from cache)")
        return _TRTModelStub(device=device)

    if efficient_backbone:
        print(
            f"Loading EfficientSAM3 ({efficient_backbone} {efficient_model}) "
            f"on {device} (resolution={imgsz})..."
        )
        model = build_efficientsam3_model(
            backbone_type=efficient_backbone,
            model_name=efficient_model,
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
        )
    else:
        skip_msg = f", skip_blocks={sorted(skip_blocks)}" if skip_blocks else ""
        if mask_blocks:
            skip_msg += f", mask_blocks={mask_blocks}"
        print(f"Loading SAM3 model on {device} (resolution={imgsz}{skip_msg})...")

        pruned_config = load_pruned_config(checkpoint_path) if checkpoint_path else None
        if pruned_config is not None:
            print(f"  Detected pruned checkpoint: {pruned_config}")
            model = build_pruned_sam3_image_model(
                checkpoint_path=checkpoint_path,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
                skip_blocks=skip_blocks,
            )
            if model.transformer.decoder.presence_token is not None:
                print("  Disabling untrained presence token for distilled checkpoint")
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint_path,
                eval_mode=True,
                skip_blocks=skip_blocks,
                mask_blocks=mask_blocks,
            )

    # Precompute position encoding buffers for non-default resolutions
    if imgsz != 1008:
        pos_enc = model.backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(imgsz)

    return model
