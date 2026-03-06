"""
Step 2 (v2): Improved Frame Registration

Two-stage registration:
  Stage 1 — Coarse: RANSAC circle fit to the outer limb (robust to the
            terminator, which is not part of the geometric circle)
  Stage 2 — Fine: phase correlation of each coarse-registered frame
            against a median reference image, correcting residual jitter
            to sub-pixel accuracy

Usage:
    python register_frames_v2.py input.mp4 --start 2 --end 10 --output registered_v2.mp4
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Phase correlation crop as a fraction of target radius
# (smaller than full radius to avoid black borders at the limb)
PHASE_CROP_RATIO = 0.89

# Number of frames to build the reference image from
REFERENCE_FRAMES = 30


def _circle_from_3pts(p1, p2, p3):
    """Unique circle through three points. Returns (cx, cy, r) or None if collinear."""
    ax, ay = p1; bx, by = p2; cx, cy = p3
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-8:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    return ux, uy, float(np.hypot(ax - ux, ay - uy))


def _fit_circle_lstsq(pts: np.ndarray):
    """Algebraic least-squares circle fit to (N,2) float64 point array."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = coeffs[0], coeffs[1]
    return cx, cy, float(np.sqrt(np.maximum(cx**2 + cy**2 + coeffs[2], 0)))


def fit_circle_ransac(contour: np.ndarray,
                      n_iter: int = 400,
                      inlier_thresh: float = 3.0) -> tuple:
    """
    RANSAC circle fit to contour points.

    The outer limb of a partially-lit moon makes up ~90% of the contour;
    the terminator (which is NOT part of the geometric circle) is the
    remaining ~10%. RANSAC treats the terminator as outliers and robustly
    recovers the true lunar circle.

    Returns (cx, cy, radius).
    """
    # Subsample for speed — 600 pts is plenty for a clean arc
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) > 600:
        idx = np.linspace(0, len(pts) - 1, 600, dtype=int)
        pts = pts[idx]

    n = len(pts)
    best_inliers = 0
    best_circle = None
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        i0, i1, i2 = rng.choice(n, 3, replace=False)
        result = _circle_from_3pts(pts[i0], pts[i1], pts[i2])
        if result is None:
            continue
        cx, cy, r = result
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        n_in = int(np.sum(np.abs(dists - r) < inlier_thresh))
        if n_in > best_inliers:
            best_inliers = n_in
            best_circle = (cx, cy, r)

    # Least-squares refinement on inliers of the best hypothesis
    cx, cy, r = best_circle
    dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    inlier_pts = pts[np.abs(dists - r) < inlier_thresh]
    if len(inlier_pts) >= 10:
        cx, cy, r = _fit_circle_lstsq(inlier_pts)

    return cx, cy, r


def detect_moon_coarse(frame: np.ndarray):
    """
    Detect moon (cx, cy, radius) using RANSAC circle fit on the outer limb.
    Robust to partial illumination — the terminator is treated as outliers.
    Returns (cx, cy, radius) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scale kernel sizes with frame dimensions (calibrated at 1080p)
    short = min(frame.shape[:2])
    blur_k = max(3, int(short * 0.057) | 1)     # ~61 at 1080p
    morph_k = max(3, int(short * 0.014) | 1)     # ~15 at 1080p
    inlier_px = max(1.0, short * 0.003)           # ~3.0 px at 1080p

    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < frame.shape[0] * frame.shape[1] * 0.01:
        return None

    cx, cy, radius = fit_circle_ransac(largest, inlier_thresh=inlier_px)
    return cx, cy, radius


def coarse_warp(frame: np.ndarray, cx: float, cy: float, radius: float,
                target_cx: int, target_cy: int, target_radius: int) -> np.ndarray:
    scale = target_radius / radius
    tx = target_cx - cx * scale
    ty = target_cy - cy * scale
    M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def extract_moon_crop(gray: np.ndarray, target_cx: int, target_cy: int,
                      phase_crop_r: int) -> np.ndarray:
    """Extract a square crop centred on the moon for phase correlation."""
    r = phase_crop_r
    y0 = target_cy - r
    y1 = target_cy + r
    x0 = target_cx - r
    x1 = target_cx + r
    return gray[y0:y1, x0:x1].astype(np.float32)


def build_reference(frames_gray: list[np.ndarray]) -> np.ndarray:
    """Median of supplied grayscale frames — suppresses transients."""
    stack = np.stack(frames_gray, axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def phase_fine_shift(crop: np.ndarray, ref_crop: np.ndarray) -> tuple[float, float]:
    """
    Sub-pixel translation of `crop` relative to `ref_crop` via phase correlation.
    Returns (dx, dy) — the shift to apply to correct the frame.
    """
    # Hanning window reduces spectral leakage from the hard crop boundary
    h, w = crop.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ref_crop * win, crop * win)
    return dx, dy


def apply_translation(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def process_video(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps) if end_sec > 0 else total_frames
    end_frame   = min(end_frame, total_frames)
    n_frames    = end_frame - start_frame

    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print(f"Processing frames {start_frame}–{end_frame} ({n_frames} frames)")

    # ------------------------------------------------------------------ #
    # Pass 1: detect moon in every frame  (no frame storage)            #
    # ------------------------------------------------------------------ #
    print("\nPass 1/3: detecting moon...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_detections = []   # (cx, cy, radius) or None  — tiny
    last_good      = None

    for _ in tqdm(range(n_frames), desc="Detecting"):
        ret, frame = cap.read()
        if not ret:
            break
        d = detect_moon_coarse(frame)
        if d is None:
            d = last_good
        else:
            last_good = d
        raw_detections.append(d)

    n_frames = len(raw_detections)

    # ------------------------------------------------------------------ #
    # Pass 1b: smooth detection parameters to suppress RANSAC noise      #
    # ------------------------------------------------------------------ #
    # Median filter over ±SMOOTH_HALF frames.  Short enough to track the
    # telescope repositioning; long enough to kill per-frame fitting noise.
    SMOOTH_HALF = 5   # ±5 frames = ±0.17 s at 30 fps
    kernel = 2 * SMOOTH_HALF + 1

    valid_mask = [d is not None for d in raw_detections]
    cxs = np.array([d[0] if d else 0.0 for d in raw_detections])
    cys = np.array([d[1] if d else 0.0 for d in raw_detections])
    rs  = np.array([d[2] if d else 0.0 for d in raw_detections])

    def median_filter_1d(arr: np.ndarray, half_win: int) -> np.ndarray:
        n = len(arr)
        out = np.empty(n)
        for i in range(n):
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            out[i] = np.median(arr[lo:hi])
        return out

    cxs = median_filter_1d(cxs, SMOOTH_HALF)
    cys = median_filter_1d(cys, SMOOTH_HALF)
    rs  = median_filter_1d(rs,  SMOOTH_HALF)

    print(f"Smoothing window: {kernel} frames ({SMOOTH_HALF/fps*1000:.0f} ms each side)")
    print(f"Smoothed radius: mean={rs.mean():.1f}  std={rs.std():.2f}")

    # Compute target geometry from video dimensions and detected moon size
    target_cx = width // 2
    target_cy = height // 2
    valid_rs = rs[np.array(valid_mask)]
    if len(valid_rs) == 0:
        sys.exit("Could not detect the moon in any frame.")
    target_radius = int(np.median(valid_rs))
    phase_crop_r = int(target_radius * PHASE_CROP_RATIO)
    # Ensure crop fits within frame
    phase_crop_r = min(phase_crop_r, target_cx, target_cy)
    print(f"Target: center=({target_cx}, {target_cy}), radius={target_radius}, "
          f"phase_crop_r={phase_crop_r}")

    # ------------------------------------------------------------------ #
    # Pass 2: build phase-correlation reference  (first 30 valid frames)#
    # ------------------------------------------------------------------ #
    print("\nPass 2/3: building reference image...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ref_crops = []
    for i in tqdm(range(min(REFERENCE_FRAMES * 2, n_frames)), desc="Reference"):
        ret, frame = cap.read()
        if not ret:
            break
        if not valid_mask[i]:
            continue
        warped = coarse_warp(frame, cxs[i], cys[i], rs[i],
                             target_cx, target_cy, target_radius)
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ref_crops.append(extract_moon_crop(gray, target_cx, target_cy, phase_crop_r))
        if len(ref_crops) >= REFERENCE_FRAMES:
            break

    if not ref_crops:
        sys.exit("Not enough valid frames to build reference.")
    reference_crop = build_reference(ref_crops)
    del ref_crops
    print(f"Reference built from {REFERENCE_FRAMES} crops.")

    # ------------------------------------------------------------------ #
    # Pass 3: coarse warp + phase correlation + write  (streaming)       #
    # ------------------------------------------------------------------ #
    print("\nPass 3/3: registering and writing...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fine_shifts = []

    for i in tqdm(range(n_frames), desc="Registering"):
        ret, frame = cap.read()
        if not ret:
            break

        if valid_mask[i]:
            warped     = coarse_warp(frame, cxs[i], cys[i], rs[i],
                                     target_cx, target_cy, target_radius)
            gray       = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            crop       = extract_moon_crop(gray, target_cx, target_cy, phase_crop_r)
            dx, dy     = phase_fine_shift(crop, reference_crop)
            frame_out  = apply_translation(warped, dx, dy)
        else:
            frame_out  = frame
            dx, dy     = 0.0, 0.0

        fine_shifts.append((dx, dy))

        cv2.circle(frame_out, (target_cx, target_cy), target_radius, (0, 255, 0), 1)
        frame_idx = start_frame + i
        ts = frame_idx / fps
        cv2.putText(frame_out,
                    f"t={ts:.2f}s  fine_shift=({dx:+.2f},{dy:+.2f})",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame_out)

    writer.release()
    cap.release()

    shifts = np.array(fine_shifts)
    print(f"\nFine shift stats:")
    print(f"  dx: mean={shifts[:,0].mean():+.2f}  std={shifts[:,0].std():.2f}  "
          f"range=[{shifts[:,0].min():+.2f}, {shifts[:,0].max():+.2f}]")
    print(f"  dy: mean={shifts[:,1].mean():+.2f}  std={shifts[:,1].std():.2f}  "
          f"range=[{shifts[:,1].min():+.2f}, {shifts[:,1].max():+.2f}]")
    print(f"Output saved to: {output_path}")

    # Write geometry sidecar for downstream scripts
    sidecar_path = os.path.splitext(output_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump({
            "moon_cx": target_cx,
            "moon_cy": target_cy,
            "moon_radius": target_radius,
        }, f, indent=2)
    print(f"Geometry: {sidecar_path}")


def main():
    parser = argparse.ArgumentParser(description="Improved frame registration v2")
    parser.add_argument("input")
    parser.add_argument("--output", default="registered.mp4")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end",   type=float, default=-1.0)
    args = parser.parse_args()
    process_video(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()

