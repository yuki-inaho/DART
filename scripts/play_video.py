#!/usr/bin/env python3
"""Loop-play a video file. Press Q or ESC to quit."""

import argparse
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="Loop-play a video (Q/ESC to quit)")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--title", default="DART demo", help="Window title")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    print(f"Playing {args.video} ({fps:.0f} fps) — press Q or ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow(args.title, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
