#!/usr/bin/env python3
"""Comprehensive FPS benchmark: all backbones x all class counts.

Runs each backbone at 1, 2, 4, 8, 16, 80 classes using 100 frames of
traffic4.mov with 10-frame warmup. Reports mean and std of per-frame
inference time (backbone + enc-dec only, no annotation/writing).
Writes results to CSV incrementally.

For 80 classes, uses trt-max-classes=16 (batched).

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_fps_benchmark.py
"""

import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.model_builder import build_sam3_image_model

VIDEO = "traffic4.mov"
WARMUP_FRAMES = 10
TIMED_FRAMES = 100
IMGSZ = 1008
CONFIDENCE = 0.3
NMS = 0.7
CSV_FILE = "benchmark_fps_results.csv"

COCO_80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

CLASS_CONFIGS = {
    1: ["person"],
    2: ["person", "car"],
    4: ["person", "car", "bicycle", "dog"],
    8: [
        "person",
        "car",
        "bicycle",
        "dog",
        "bus",
        "truck",
        "motorcycle",
        "traffic light",
    ],
    16: COCO_80[:16],
    80: COCO_80,
}

BACKBONES = [
    {
        "name": "ViT-H (full)",
        "engine": "hf_backbone_1008_fp16_opt5.engine",
        "fallback_engine": "hf_backbone_1008_fp16.engine",
    },
    {
        "name": "ViT-H Pruned-16",
        "engine": "hf_backbone_1008_pruned_fp16_opt5.engine",
        "fallback_engine": "hf_backbone_1008_pruned_fp16.engine",
    },
    {
        "name": "RepViT-M2.3",
        "engine": "student_repvit_m2_3_fp16_opt5.engine",
        "fallback_engine": "student_repvit_m2_3_fp16.engine",
    },
    {
        "name": "TinyViT-21M",
        "engine": "student_tiny_vit_21m_fp16_opt5.engine",
        "fallback_engine": "student_tiny_vit_21m_fp16.engine",
    },
    {
        "name": "EfficientViT-L2",
        "engine": "student_efficientvit_l2_fp16_opt5.engine",
        "fallback_engine": "student_efficientvit_l2_fp16.engine",
    },
    {
        "name": "EfficientViT-L1",
        "engine": "student_efficientvit_l1_fp16_opt5.engine",
        "fallback_engine": "student_efficientvit_l1_fp16.engine",
    },
]


def get_enc_dec_config(num_classes):
    """Return (engine_path, trt_max_classes) for given class count."""
    configs = [
        (1, "enc_dec_1008_c1_presence_fp16_opt5.engine"),
        (2, "enc_dec_1008_c2_presence_fp16_opt5.engine"),
        (4, "enc_dec_1008_c4_presence_fp16_opt5.engine"),
        (8, "enc_dec_1008_c8_presence_fp16_opt5.engine"),
        (16, "enc_dec_1008_c16_presence_fp16_opt5.engine"),
    ]
    for max_c, engine in configs:
        if num_classes <= max_c:
            return engine, max_c
    # 80 classes: use 16-class engine in batched mode
    return "enc_dec_1008_c16_presence_fp16_opt5.engine", 16


def resolve_engine(backbone):
    """Find the engine file, trying opt5 first, then fallback."""
    if os.path.exists(backbone["engine"]):
        return backbone["engine"]
    fb = backbone.get("fallback_engine", "")
    if fb and os.path.exists(fb):
        print(f"  Using fallback engine: {fb}")
        return fb
    return None


def load_frames(video_path, n_frames):
    """Load n_frames from a video file as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while len(frames) < n_frames:
        ret, bgr = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ret, bgr = cap.read()
            if not ret:
                break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


@torch.inference_mode()
def benchmark_one(predictor, frames, warmup_frames, classes):
    """Run benchmark, return (bb_times, ed_times, total_times) in ms."""
    bb_times = []
    ed_times = []
    total_times = []

    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        state = predictor.set_image(frame)
        torch.cuda.synchronize()
        t_bb = time.perf_counter()

        results = predictor.predict(
            state,
            confidence_threshold=CONFIDENCE,
            nms_threshold=NMS,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        if i >= warmup_frames:
            bb_ms = (t_bb - t0) * 1000
            ed_ms = (t_end - t_bb) * 1000
            total_ms = (t_end - t0) * 1000
            bb_times.append(bb_ms)
            ed_times.append(ed_ms)
            total_times.append(total_ms)

        if i % 20 == 0:
            n_dets = len(results["scores"])
            ms = (t_end - t0) * 1000
            phase = "warmup" if i < warmup_frames else "timed"
            print(f"    Frame {i:3d} [{phase}]: {ms:.1f}ms, {n_dets} dets")

    return np.array(bb_times), np.array(ed_times), np.array(total_times)


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Video: {VIDEO}")
    print(f"Resolution: {IMGSZ}")
    print(f"Warmup: {WARMUP_FRAMES} frames, Timed: {TIMED_FRAMES} frames")
    print()

    # Load frames once
    total_needed = WARMUP_FRAMES + TIMED_FRAMES
    print(f"Loading {total_needed} frames from {VIDEO} ...")
    frames = load_frames(VIDEO, total_needed)
    print(f"  Loaded {len(frames)} frames")

    # Load SAM3 model once (shared across all backbones)
    print("Loading SAM3 model ...")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path="sam3.pt",
        eval_mode=True,
    )

    # CSV setup
    fieldnames = [
        "backbone",
        "num_classes",
        "bb_mean_ms",
        "bb_std_ms",
        "ed_mean_ms",
        "ed_std_ms",
        "total_mean_ms",
        "total_std_ms",
        "fps_mean",
        "n_frames",
    ]
    write_header = not os.path.exists(CSV_FILE)
    csvf = open(CSV_FILE, "a", newline="")
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csvf.flush()

    total_runs = len(BACKBONES) * len(CLASS_CONFIGS)
    run_idx = 0

    for backbone in BACKBONES:
        engine_path = resolve_engine(backbone)
        if engine_path is None:
            print(f"\n  SKIP {backbone['name']}: no engine found")
            for nc in CLASS_CONFIGS:
                run_idx += 1
                writer.writerow(
                    {
                        "backbone": backbone["name"],
                        "num_classes": nc,
                        **dict.fromkeys(fieldnames[2:], "N/A"),
                    }
                )
            csvf.flush()
            continue

        for num_classes in [1, 2, 4, 8, 16, 80]:
            run_idx += 1
            classes = CLASS_CONFIGS[num_classes]
            enc_dec_engine, trt_max_classes = get_enc_dec_config(num_classes)

            print(f"\n{'=' * 70}")
            print(
                f"  [{run_idx}/{total_runs}] {backbone['name']} @ {num_classes} classes"
            )
            print(f"  Backbone: {engine_path}")
            print(f"  Enc-dec:  {enc_dec_engine} (max_classes={trt_max_classes})")
            print(f"{'=' * 70}")

            if not os.path.exists(enc_dec_engine):
                print(f"  SKIP: {enc_dec_engine} not found")
                writer.writerow(
                    {
                        "backbone": backbone["name"],
                        "num_classes": num_classes,
                        **dict.fromkeys(fieldnames[2:], "N/A"),
                    }
                )
                csvf.flush()
                continue

            # Create predictor with this backbone engine
            predictor = Sam3MultiClassPredictorFast(
                model,
                device=device,
                resolution=IMGSZ,
                trt_engine_path=engine_path,
                use_fp16=True,
                detection_only=True,
                trt_enc_dec_engine_path=enc_dec_engine,
                trt_max_classes=trt_max_classes,
            )

            print(f"  Setting {num_classes} classes ...")
            predictor.set_classes(classes)

            # Warmup the predictor
            print("  Warming up ...")
            dummy = Image.new("RGB", (IMGSZ, IMGSZ))
            for _ in range(3):
                state = predictor.set_image(dummy)
                predictor.predict(state, confidence_threshold=0.5)
            torch.cuda.synchronize()

            # Run benchmark
            print(f"  Running {WARMUP_FRAMES}+{TIMED_FRAMES} frames ...")
            bb_times, ed_times, total_times = benchmark_one(
                predictor,
                frames,
                WARMUP_FRAMES,
                classes,
            )

            # Stats
            row = {
                "backbone": backbone["name"],
                "num_classes": num_classes,
                "bb_mean_ms": f"{bb_times.mean():.1f}",
                "bb_std_ms": f"{bb_times.std():.1f}",
                "ed_mean_ms": f"{ed_times.mean():.1f}",
                "ed_std_ms": f"{ed_times.std():.1f}",
                "total_mean_ms": f"{total_times.mean():.1f}",
                "total_std_ms": f"{total_times.std():.1f}",
                "fps_mean": f"{1000.0 / total_times.mean():.1f}",
                "n_frames": len(total_times),
            }
            writer.writerow(row)
            csvf.flush()

            print(f"\n  >> {backbone['name']} @ {num_classes}cls:")
            print(f"     BB:    {bb_times.mean():.1f} +/- {bb_times.std():.1f} ms")
            print(f"     E-D:   {ed_times.mean():.1f} +/- {ed_times.std():.1f} ms")
            print(
                f"     Total: {total_times.mean():.1f} +/- {total_times.std():.1f} ms"
            )
            print(f"     FPS:   {1000.0 / total_times.mean():.1f}")

            # Cleanup predictor to free GPU memory
            del predictor
            torch.cuda.empty_cache()

    csvf.close()

    # Print summary table
    print(f"\n\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE — results in {CSV_FILE}")
    print(f"{'=' * 70}")

    # Read and print CSV
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(
        f"\n{'Backbone':<20} {'#Cls':>5} {'BB(ms)':>10} {'E-D(ms)':>10} "
        f"{'Total(ms)':>12} {'FPS':>8}"
    )
    print(f"{'-' * 20} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 8}")
    for r in rows:
        bb = r["bb_mean_ms"]
        ed = r["ed_mean_ms"]
        total = r["total_mean_ms"]
        total_std = r["total_std_ms"]
        fps = r["fps_mean"]
        if total != "N/A":
            total_str = f"{total}+/-{total_std}"
        else:
            total_str = "N/A"
        print(
            f"{r['backbone']:<20} {r['num_classes']:>5} {bb:>10} {ed:>10} "
            f"{total_str:>12} {fps:>8}"
        )


if __name__ == "__main__":
    main()
