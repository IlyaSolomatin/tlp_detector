"""
Make a review video from events.json + detections.mp4.

For each event:
  - Extracts a clip: 2 s before event start → 2 s after event start
  - Crops a 10× zoom centred on the event position
  - Prepends a short title card

All clips are concatenated into one continuous review video.

Usage:
    python make_events_video.py
    python make_events_video.py --events events.json --input detections.mp4 --output review.mp4
    python make_events_video.py --pad 3 --zoom 8
"""

import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm

# Output frame size (zoom crops are scaled up to this)
OUT_W, OUT_H = 960, 540


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        events = json.load(f)
    # Sort chronologically
    return sorted(events, key=lambda e: e["start_frame"])


def make_title_card(event: dict, fps: float, n_frames: int) -> list[np.ndarray]:
    """A short black title card shown before each event clip."""
    card = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    lines = [
        f"Event #{event['event_id']}",
        f"t = {event['start_time_s']:.2f}s  –  {event['end_time_s']:.2f}s",
        f"Duration: {event['duration_frames']} frames",
        f"Peak SNR: {event['peak_snr']:.1f}",
        f"Lunar pos: ({event['lunar_x']:+.0f}, {event['lunar_y']:+.0f}) px",
    ]
    y = OUT_H // 2 - len(lines) * 22
    for i, line in enumerate(lines):
        cv2.putText(card, line, (OUT_W // 2 - 220, y + i * 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    # Hold for ~1 second
    card_frames = max(1, int(fps))
    return [card] * card_frames


def zoom_crop(frame: np.ndarray, cx: float, cy: float, zoom: int) -> np.ndarray:
    """Crop zoom× region centred on (cx, cy), scale to OUT_W × OUT_H."""
    h, w = frame.shape[:2]
    crop_w = w // zoom
    crop_h = h // zoom

    x0 = int(cx) - crop_w // 2
    y0 = int(cy) - crop_h // 2
    # Clamp to frame boundaries
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))

    crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


def process(
    input_path: str,
    events_path: str,
    output_path: str,
    pad_sec: float,
    zoom: int,
) -> None:
    events = load_events(events_path)
    if not events:
        print("No events in JSON — nothing to render.")
        return

    cap = cv2.VideoCapture(input_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pad_frames = int(pad_sec * fps)

    print(f"Source : {src_w}x{src_h} @ {fps:.2f} fps, {total} frames")
    print(f"Events : {len(events)}")
    print(f"Pad    : ±{pad_sec}s ({pad_frames} frames each side)")
    print(f"Zoom   : {zoom}×  →  crop {src_w//zoom}x{src_h//zoom} → {OUT_W}x{OUT_H}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

    for event in tqdm(events, desc="Events"):
        cx = event["cx"]
        cy = event["cy"]

        clip_start = max(0,     event["start_frame"] - pad_frames)
        clip_end   = min(total, event["start_frame"] + pad_frames)

        # Seek and read clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for fi in range(clip_start, clip_end):
            ret, frame = cap.read()
            if not ret:
                break

            out_frame = zoom_crop(frame, cx, cy, zoom)

            # Timestamp overlay
            ts = fi / fps
            is_event = event["start_frame"] <= fi <= event["end_frame"]
            color = (0, 0, 255) if is_event else (180, 180, 180)
            cv2.putText(out_frame,
                        f"#{event['event_id']}  t={ts:.3f}s  frame={fi}",
                        (10, OUT_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build per-event review video")
    parser.add_argument("--events", default="events.json")
    parser.add_argument("--input",  default="detections.mp4",
                        help="Annotated video (output of run_pipeline.py)")
    parser.add_argument("--output", default="review.mp4")
    parser.add_argument("--pad",    type=float, default=2.0,
                        help="Seconds of context before/after event start (default 2)")
    parser.add_argument("--zoom",   type=int,   default=10,
                        help="Zoom factor (default 10)")
    args = parser.parse_args()

    process(args.input, args.events, args.output, args.pad, args.zoom)


if __name__ == "__main__":
    main()
