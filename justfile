# DART demo commands
# Usage: just demo-image ~/Pictures/apple.png apple
#        just demo-video ~/Videos/tomato.mkv tomato

default_imgsz := "504"
default_max_frames := "30"
default_fps := "0"
default_checkpoint := "models/sam3.pt"
python := ".venv/bin/python"

# Single image detection (opens annotated result)
demo-image path classes="apple":
    {{python}} demo_multiclass.py \
        --image {{path}} \
        --classes {{classes}} \
        --checkpoint {{default_checkpoint}} \
        --detection-only \
        --imgsz {{default_imgsz}} \
        -o /tmp/dart_demo.jpg
    xdg-open /tmp/dart_demo.jpg 2>/dev/null || echo "Saved to /tmp/dart_demo.jpg"

# Video detection: detect with tqdm → loop playback (Q/ESC to quit)
demo-video path classes="tomato" max_frames=default_max_frames imgsz=default_imgsz fps=default_fps:
    {{python}} demo_video.py \
        --video {{path}} \
        --classes {{classes}} \
        --checkpoint {{default_checkpoint}} \
        --compile default \
        --imgsz {{imgsz}} \
        --max-frames {{max_frames}} \
        --fps {{fps}} \
        -o /tmp/dart_demo_video.mp4
    {{python}} scripts/play_video.py /tmp/dart_demo_video.mp4

# Video detection → save to file only (no playback)
demo-video-save path output classes="tomato" max_frames=default_max_frames imgsz=default_imgsz fps=default_fps:
    {{python}} demo_video.py \
        --video {{path}} \
        --classes {{classes}} \
        --checkpoint {{default_checkpoint}} \
        --compile default \
        --imgsz {{imgsz}} \
        --max-frames {{max_frames}} \
        --fps {{fps}} \
        -o {{output}}

# Run regression tests
test:
    {{python}} -m pytest tests/ -v

# Lint check
lint:
    {{python}} -m ruff check

# Type check
typecheck:
    {{python}} -m ty check

# Format code
fmt:
    {{python}} -m ruff format
    {{python}} -m ruff check --fix

# Complexity report (C+ grade functions)
complexity:
    {{python}} -m radon cc sam3/ demo_multiclass.py demo_video.py -n C -s
