#!/usr/bin/env python3
"""Build all TRT engines needed for the comprehensive FPS benchmark.

Builds:
  - enc-dec with presence for 1, 2, 4, 8, 16 classes at FP16 opt5
  - Full ViT-H backbone at FP16 opt5
  - Pruned-16 backbone at FP16 opt5
  - Student backbones (EfficientViT-L1, L2, TinyViT, RepViT) at FP16 opt5
"""

import os
import subprocess
import sys
import time

PYTHON = sys.executable


def run(cmd, desc):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env, timeout=1200)
    dt = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  {status} in {dt:.0f}s")
    return result.returncode == 0


def main():
    # 1. Build enc-dec engines with presence token
    for nc in [1, 2, 4, 8, 16]:
        onnx = f"enc_dec_1008_c{nc}_presence.onnx"
        engine = f"enc_dec_1008_c{nc}_presence_fp16_opt5.engine"
        if os.path.exists(engine):
            print(f"  Skipping {engine} (already exists)")
            continue
        run(
            [
                PYTHON,
                "-m",
                "sam3.trt.build_engine",
                "--onnx",
                onnx,
                "--output",
                engine,
                "--fp16",
                "--mixed-precision",
                "none",
                "--opt-level",
                "5",
            ],
            f"Enc-dec c{nc} presence FP16 opt5",
        )

    # 2. Full ViT-H backbone
    # Re-export + build via HF path
    hf_engine = "hf_backbone_1008_fp16_opt5.engine"
    if not os.path.exists(hf_engine):
        hf_onnx = "onnx_hf_backbone_1008/hf_backbone.onnx"
        if os.path.exists(hf_onnx):
            run(
                [
                    PYTHON,
                    "-m",
                    "sam3.trt.build_engine",
                    "--onnx",
                    hf_onnx,
                    "--output",
                    hf_engine,
                    "--fp16",
                    "--mixed-precision",
                    "none",
                    "--opt-level",
                    "5",
                ],
                "Full ViT-H backbone FP16 opt5",
            )
        else:
            print(f"  WARNING: {hf_onnx} not found, skipping full backbone")
    else:
        print(f"  Skipping {hf_engine} (already exists)")

    # 3. Pruned-16 backbone
    pruned_engine = "hf_backbone_1008_pruned_fp16_opt5.engine"
    if not os.path.exists(pruned_engine):
        # Use existing pruned ONNX from HF export path
        pruned_onnx = "onnx_hf_backbone_1008_pruned/hf_backbone.onnx"
        if not os.path.exists(pruned_onnx):
            # Try the other path
            pruned_onnx = "pruned_backbone_1008.onnx"
        if os.path.exists(pruned_onnx):
            run(
                [
                    PYTHON,
                    "-m",
                    "sam3.trt.build_engine",
                    "--onnx",
                    pruned_onnx,
                    "--output",
                    pruned_engine,
                    "--fp16",
                    "--mixed-precision",
                    "none",
                    "--opt-level",
                    "5",
                ],
                "Pruned-16 backbone FP16 opt5",
            )
        else:
            print("  WARNING: No pruned ONNX found, skipping")
    else:
        print(f"  Skipping {pruned_engine} (already exists)")

    # 4. Student backbones
    students = [
        (
            "student_efficientvit_l1_fixed.onnx",
            "student_efficientvit_l1_fp16_opt5.engine",
            "EfficientViT-L1",
        ),
        (
            "student_efficientvit_l2.onnx",
            "student_efficientvit_l2_fp16_opt5.engine",
            "EfficientViT-L2",
        ),
        (
            "student_tiny_vit_21m.onnx",
            "student_tiny_vit_21m_fp16_opt5.engine",
            "TinyViT-21M",
        ),
        (
            "student_repvit_m2_3.onnx",
            "student_repvit_m2_3_fp16_opt5.engine",
            "RepViT-M2.3",
        ),
    ]
    for onnx, engine, name in students:
        if os.path.exists(engine):
            print(f"  Skipping {engine} (already exists)")
            continue
        if not os.path.exists(onnx):
            print(f"  WARNING: {onnx} not found, skipping {name}")
            continue
        run(
            [
                PYTHON,
                "-m",
                "sam3.trt.build_engine",
                "--onnx",
                onnx,
                "--output",
                engine,
                "--fp16",
                "--mixed-precision",
                "none",
                "--opt-level",
                "5",
            ],
            f"{name} backbone FP16 opt5",
        )

    print(f"\n{'=' * 60}")
    print("  ALL ENGINE BUILDS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
