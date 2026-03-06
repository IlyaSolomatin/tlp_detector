"""
Step 3: Background Estimation & Subtraction

Takes the registered video, computes a rolling temporal median background
over a ±W frame window, and subtracts it from each frame.

Outputs a side-by-side video:
  Left  — registered frame (colour)
  Right — residual (frame − background), amplified for visibility

The lunar disk mask is also computed and overlaid so we can see which
region will be analysed in later steps.

Usage:
    python background_subtract.py registered.mp4 --output residual.mp4 --half-window 15
"""

import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Fixed moon position (must match register_frames.py)
MOON_CX = 960
MOON_CY = 540
MOON_RADIUS = 426
# Erode inward to avoid limb artifacts (~5% of radius)
MASK_RADIUS = int(MOON_RADIUS * 0.95)


def build_disk_mask(height: int, width: int) -> np.ndarray:
    """Boolean mask — True inside the analysis region of the lunar disk."""
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - MOON_CX) ** 2 + (Y - MOON_CY) ** 2)
    return dist <= MASK_RADIUS


def load_frames(cap: cv2.VideoCapture, start_frame: int, n_frames: int):
    """
    Load all frames in the range as grayscale (for background) and colour
    (for output). Returns (gray_array, color_list).
    gray_array shape: (n_frames, H, W), dtype uint8
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    gray_frames = []
    color_frames = []

    for _ in tqdm(range(n_frames), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        color_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    gray_array = np.array(gray_frames, dtype=np.uint8)  # (N, H, W)
    return gray_array, color_frames


def make_residual_display(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convert a signed float residual into a displayable uint8 image.
    Positive residuals (brighter than background) → warm colours.
    Outside the mask → dimmed.
    """
    # Centre at 128: 0 = no change, >128 = brighter, <128 = darker
    amplified = np.clip(residual * 4 + 128, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)

    # Dim the area outside the analysis mask
    outside = ~mask
    bgr[outside] = (bgr[outside] * 0.3).astype(np.uint8)

    # Draw the mask boundary
    cv2.circle(bgr, (MOON_CX, MOON_CY), MASK_RADIUS, (0, 200, 0), 1)

    return bgr


def process(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    half_window: int,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec > 0 else total_frames
    end_frame = min(end_frame, total_frames)
    n_frames = end_frame - start_frame

    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print(f"Processing frames {start_frame}–{end_frame} ({n_frames} frames)")
    print(f"Half-window W={half_window} ({half_window/fps:.2f}s each side)")
    estimated_mb = n_frames * height * width / 1e6
    print(f"Estimated grayscale buffer: {estimated_mb:.0f} MB")

    gray_array, color_frames = load_frames(cap, start_frame, n_frames)
    cap.release()

    n_frames = len(color_frames)  # actual frames read
    mask = build_disk_mask(height, width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nComputing backgrounds and writing output...")
    residual_peaks = []

    for t in tqdm(range(n_frames), desc="Background subtraction"):
        lo = max(0, t - half_window)
        hi = min(n_frames, t + half_window + 1)

        # Temporal median of the window (excluding current frame to avoid
        # the flash suppressing itself — matters most for single-frame events)
        window = np.concatenate([gray_array[lo:t], gray_array[t+1:hi]], axis=0)
        if len(window) == 0:
            window = gray_array[lo:hi]  # fallback at very start/end
        background = np.median(window, axis=0).astype(np.float32)

        current = gray_array[t].astype(np.float32)
        residual = current - background  # signed, float32

        # Track peak positive residual inside the mask for diagnostics
        peak = float(residual[mask].max()) if mask.any() else 0.0
        residual_peaks.append(peak)

        # Full-resolution residual visualisation
        out_frame = make_residual_display(residual, mask)

        frame_idx = start_frame + t
        ts = frame_idx / fps
        label = f"t={ts:.2f}s | W={half_window} | peak={peak:.1f}"
        cv2.putText(out_frame, label, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(out_frame, "RESIDUAL (x4)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        writer.write(out_frame)

    writer.release()

    peaks = np.array(residual_peaks)
    print(f"\nResidual peak stats (inside mask, positive only):")
    print(f"  mean={peaks.mean():.1f}  std={peaks.std():.1f}  "
          f"max={peaks.max():.1f}  p99={np.percentile(peaks, 99):.1f}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Background subtraction — Step 3 of TLP pipeline")
    parser.add_argument("input", help="Registered video (output of register_frames.py)")
    parser.add_argument("--output", default="residual.mp4")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=-1.0)
    parser.add_argument("--half-window", type=int, default=15,
                        help="Frames each side of current for median (default 15 = ±0.5s)")
    args = parser.parse_args()

    process(args.input, args.output, args.start, args.end, args.half_window)


if __name__ == "__main__":
    main()
