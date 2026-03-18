"""
Step 4: Flash Detection

Takes the registered video, computes per-frame residuals (temporal median
background), thresholds, filters spatially, then clusters across frames
to form candidate events.

Output:
  - Annotated video with candidate detections highlighted
  - events.json with per-event metadata

Usage:
    python detect_flashes.py registered.mp4 --output detections.mp4
    python detect_flashes.py registered.mp4 --sigma-k 5 --min-frames 2
"""

import argparse
import json
import os
import sys
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

MASK_EROSION = 0.95  # erode inward to avoid limb artefacts


def load_geometry(video_path: str) -> tuple[int, int, int]:
    """Load moon geometry from the sidecar JSON written by register_frames.py."""
    sidecar = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(sidecar):
        sys.exit(
            f"Geometry sidecar not found: {sidecar}\n"
            f"Run register_frames.py first to generate it, or pass "
            f"--moon-cx, --moon-cy, --moon-radius manually."
        )
    with open(sidecar) as f:
        geo = json.load(f)
    return int(geo["moon_cx"]), int(geo["moon_cy"]), int(geo["moon_radius"])

# ── Detection defaults ──────────────────────────────────────────────────────
DEFAULT_HALF_WINDOW  = 15    # frames each side for temporal median
DEFAULT_SIGMA_K      = 5.0   # detection threshold in σ units
DEFAULT_MIN_RESIDUAL = 30.0  # hard floor in DN — must exceed jitter ceiling
DEFAULT_MIN_AREA     = 2     # pixels — kills single-pixel noise
DEFAULT_MAX_AREA     = 150   # pixels — kills large artefacts / moving objects
DEFAULT_MIN_FRAMES   = 2     # consecutive frames to confirm an event
DEFAULT_MAX_FRAMES   = 30    # frames — events longer than this are not flashes
DEFAULT_LINK_RADIUS  = 12    # px — max spatial distance to link detections


@dataclass
class Detection:
    frame_idx: int
    cx: float
    cy: float
    peak_snr: float    # peak residual / sigma
    area: int          # pixels above threshold


@dataclass
class Event:
    event_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    cx: float          # mean centroid x
    cy: float          # mean centroid y
    peak_snr: float    # max peak_snr across all detections
    detections: list   # list of Detection dicts


def build_disk_mask(height: int, width: int,
                    moon_cx: int, moon_cy: int, mask_radius: int) -> np.ndarray:
    Y, X = np.ogrid[:height, :width]
    return (X - moon_cx) ** 2 + (Y - moon_cy) ** 2 <= mask_radius ** 2


def robust_sigma(residual: np.ndarray, mask: np.ndarray) -> float:
    """
    Noise estimate inside the disk mask.

    We use the full standard deviation of the residual within the mask.
    This naturally includes the jitter contribution: on jitter-heavy frames
    σ rises, making k·σ a meaningful barrier above the worst jitter artefacts.
    On quiet frames σ falls, giving more sensitivity. A floor of 1.0 DN
    prevents over-triggering on frames that have been perfectly subtracted.
    """
    vals = residual[mask].astype(np.float32)
    return max(float(np.std(vals)), 1.0)


def detect_in_frame(
    residual: np.ndarray,
    sigma: float,
    mask: np.ndarray,
    k: float,
    min_residual: float,
    min_area: int,
    max_area: int,
) -> list[Detection]:
    """Return candidate detections in a single residual frame."""
    if sigma < 1e-6:
        return []

    # Both conditions must hold: relative (k·σ) AND absolute (min_residual).
    # The absolute floor prevents over-triggering when sigma is small (quiet frames).
    threshold = max(k * sigma, min_residual)
    binary = ((residual > threshold) & mask).astype(np.uint8)

    # Small morphological opening: remove isolated hot pixels
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               np.ones((2, 2), np.uint8))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    detections = []
    for label in range(1, n_labels):   # skip background label 0
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        # Circularity filter: rejects elongated jitter arcs (crater-edge ghosts).
        # Circularity = 4π·area / perimeter²  →  1.0 for a perfect circle,
        # much lower for arcs/crescents. Real flashes are compact blobs ≥ 0.3.
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        if circularity < 0.3:
            continue

        cx, cy = centroids[label]
        peak_val = float(residual[component_mask.astype(bool)].max())
        peak_snr = peak_val / sigma
        detections.append(Detection(
            frame_idx=-1,   # filled by caller
            cx=cx,
            cy=cy,
            peak_snr=peak_snr,
            area=area,
        ))
    return detections


def cluster_detections(
    all_detections: list[Detection],
    link_radius: float,
    min_frames: int,
    max_frames: int,
) -> list[Event]:
    """
    Simple greedy frame-by-frame linker.
    Each unmatched detection starts a new track; existing tracks are
    extended if a detection falls within link_radius pixels.
    """
    if not all_detections:
        return []

    # Sort by frame
    dets = sorted(all_detections, key=lambda d: d.frame_idx)

    # Active tracks: list of lists of Detection
    active: list[list[Detection]] = []
    closed: list[list[Detection]] = []

    for det in dets:
        matched = False
        for track in active:
            last = track[-1]
            # Must be in the next few frames
            if det.frame_idx - last.frame_idx > 3:
                continue
            dist = np.hypot(det.cx - last.cx, det.cy - last.cy)
            if dist <= link_radius:
                track.append(det)
                matched = True
                break
        if not matched:
            active.append([det])

        # Close tracks that haven't been updated
        still_active = []
        for track in active:
            if det.frame_idx - track[-1].frame_idx > 3:
                closed.append(track)
            else:
                still_active.append(track)
        active = still_active

    closed.extend(active)

    events = []
    event_id = 0
    for track in closed:
        duration = track[-1].frame_idx - track[0].frame_idx + 1
        if duration < min_frames or duration > max_frames:
            continue
        event_id += 1
        events.append(Event(
            event_id=event_id,
            start_frame=track[0].frame_idx,
            end_frame=track[-1].frame_idx,
            duration_frames=duration,
            cx=float(np.mean([d.cx for d in track])),
            cy=float(np.mean([d.cy for d in track])),
            peak_snr=float(max(d.peak_snr for d in track)),
            detections=[asdict(d) for d in track],
        ))

    # Sort by peak SNR descending
    events.sort(key=lambda e: e.peak_snr, reverse=True)
    return events


def process(
    input_path: str,
    output_path: str,
    events_path: str,
    start_sec: float,
    end_sec: float,
    half_window: int,
    sigma_k: float,
    min_residual: float,
    min_area: int,
    max_area: int,
    min_frames: int,
    max_frames: int,
    link_radius: float,
    moon_cx: int | None = None,
    moon_cy: int | None = None,
    moon_radius: int | None = None,
) -> list[Event]:
    if not os.path.isfile(input_path):
        sys.exit(f"Input file not found: {input_path}")
    # Resolve moon geometry
    if moon_cx is None or moon_cy is None or moon_radius is None:
        moon_cx, moon_cy, moon_radius = load_geometry(input_path)
    mask_radius = int(moon_radius * MASK_EROSION)
    print(f"Moon geometry: center=({moon_cx}, {moon_cy}), "
          f"radius={moon_radius}, mask_radius={mask_radius}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {input_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps) if end_sec > 0 else n_total
    end_frame   = min(end_frame, n_total)
    n_frames    = end_frame - start_frame

    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print(f"Frames {start_frame}–{end_frame} ({n_frames} frames, "
          f"{n_frames/fps:.1f}s)")
    print(f"σ threshold: k={sigma_k}, min_residual={min_residual} DN  |  "
          f"size: [{min_area}, {max_area}]px  |  "
          f"duration: [{min_frames}, {max_frames}] frames")

    mask = build_disk_mask(height, width, moon_cx, moon_cy, mask_radius)

    # ── Pass 1: sliding-window detection  (grayscale only, bounded memory) ─
    # Buffer holds at most 2*W+1 grayscale frames at any time.
    # For a 174 s video at 30 fps with W=15: max 31 frames × 2 MB = 62 MB.
    W = half_window
    gray_buf: deque[np.ndarray] = deque()
    buf_lo   = 0   # clip-relative index (0-based) of gray_buf[0]
    frames_read = 0

    all_detections: list[Detection] = []
    sigma_history:  list[float]     = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for t in tqdm(range(n_frames), desc="Pass 1/2 — detecting"):
        # Fill lookahead: read frames until we have frame t+W in the buffer
        while frames_read < min(t + W + 1, n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frames_read += 1

        t_in_buf = t - buf_lo
        if t_in_buf < 0 or t_in_buf >= len(gray_buf):
            continue

        # Window: frames [max(0,t-W) .. min(n,t+W)] excluding t itself
        lo_buf = max(0, t - W) - buf_lo
        hi_buf = min(n_frames, t + W + 1) - buf_lo
        window = [gray_buf[j]
                  for j in range(max(0, lo_buf), min(len(gray_buf), hi_buf))
                  if j != t_in_buf]

        if window:
            background = np.median(np.stack(window, axis=0), axis=0).astype(np.float32)
        else:
            background = gray_buf[t_in_buf].astype(np.float32)

        residual = gray_buf[t_in_buf].astype(np.float32) - background
        sigma    = robust_sigma(residual, mask)
        sigma_history.append(sigma)

        dets = detect_in_frame(residual, sigma, mask,
                               sigma_k, min_residual, min_area, max_area)
        for d in dets:
            d.frame_idx = start_frame + t
        all_detections.extend(dets)

        # Evict frames that are no longer needed as background for future frames
        while buf_lo < t - W:
            gray_buf.popleft()
            buf_lo += 1

    cap.release()

    print(f"\nRaw detections: {len(all_detections)} across {n_frames} frames")
    print(f"Sigma: mean={np.mean(sigma_history):.2f}  "
          f"std={np.std(sigma_history):.2f}  "
          f"range=[{np.min(sigma_history):.2f}, {np.max(sigma_history):.2f}]")

    # ── Temporal clustering ────────────────────────────────────────────────
    events = cluster_detections(all_detections, link_radius,
                                min_frames, max_frames)
    print(f"Events after clustering: {len(events)}")

    # ── Pass 2: annotate and write  (one color frame at a time)  ──────────
    frame_events: dict[int, list[Event]] = {}
    for ev in events:
        for fi in range(ev.start_frame, ev.end_frame + 1):
            frame_events.setdefault(fi, []).append(ev)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for t in tqdm(range(n_frames), desc="Pass 2/2 — writing"):
        ret, frame = cap.read()
        if not ret:
            break

        fi  = start_frame + t
        ts  = fi / fps
        out = frame

        cv2.circle(out, (moon_cx, moon_cy), mask_radius, (40, 40, 40), 1)

        if fi in frame_events:
            marker_r = max(5, moon_radius // 21)   # ~20 px at radius 426
            for ev in frame_events[fi]:
                cx, cy = int(ev.cx), int(ev.cy)
                cv2.circle(out, (cx, cy), marker_r, (0, 0, 255), 2)
                label = f"#{ev.event_id} SNR={ev.peak_snr:.1f}"
                cv2.putText(out, label, (cx + marker_r + 4, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(out, f"t={ts:.2f}s  k={sigma_k}",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        writer.write(out)

    cap.release()
    writer.release()

    # ── Save events JSON ───────────────────────────────────────────────────
    events_data = []
    for ev in events:
        events_data.append({
            **asdict(ev),
            "start_time_s": ev.start_frame / fps,
            "end_time_s":   ev.end_frame   / fps,
            "lunar_x": ev.cx - moon_cx,
            "lunar_y": ev.cy - moon_cy,
        })
    with open(events_path, "w") as f:
        json.dump(events_data, f, indent=2)

    # ── Console summary ────────────────────────────────────────────────────
    if events:
        print(f"\n{'#':>3}  {'frames':>10}  {'time (s)':>12}  "
              f"{'dur':>5}  {'SNR':>6}  {'lunar (x,y)':>14}")
        print("─" * 60)
        for ev in events:
            t0 = ev.start_frame / fps
            t1 = ev.end_frame   / fps
            lx = ev.cx - moon_cx
            ly = ev.cy - moon_cy
            print(f"{ev.event_id:>3}  "
                  f"{ev.start_frame:>5}–{ev.end_frame:<5}  "
                  f"{t0:>5.2f}–{t1:<5.2f}  "
                  f"{ev.duration_frames:>5}  "
                  f"{ev.peak_snr:>6.1f}  "
                  f"({lx:+.0f},{ly:+.0f})")
    else:
        print("\nNo events detected. Try lowering --sigma-k.")

    print(f"\nAnnotated video: {output_path}")
    print(f"Events JSON:     {events_path}")
    return events


def main():
    parser = argparse.ArgumentParser(description="Flash detector — Step 4 of TLP pipeline")
    parser.add_argument("input", help="Registered video (output of register_frames.py)")
    parser.add_argument("--output",      default="detections.mp4")
    parser.add_argument("--events",      default="events.json")
    parser.add_argument("--start",       type=float, default=0.0)
    parser.add_argument("--end",         type=float, default=-1.0)
    parser.add_argument("--half-window", type=int,   default=DEFAULT_HALF_WINDOW)
    parser.add_argument("--sigma-k",       type=float, default=DEFAULT_SIGMA_K,
                        help="Detection threshold in σ units (default 5)")
    parser.add_argument("--min-residual", type=float, default=DEFAULT_MIN_RESIDUAL,
                        help="Hard minimum residual in DN — must exceed this regardless of sigma "
                             "(default 30, set lower to be more sensitive)")
    parser.add_argument("--min-area",    type=int,   default=DEFAULT_MIN_AREA)
    parser.add_argument("--max-area",    type=int,   default=DEFAULT_MAX_AREA)
    parser.add_argument("--min-frames",  type=int,   default=DEFAULT_MIN_FRAMES,
                        help="Min consecutive frames for a valid event (default 2)")
    parser.add_argument("--max-frames",  type=int,   default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--link-radius", type=float, default=DEFAULT_LINK_RADIUS)
    parser.add_argument("--moon-cx",     type=int, default=None,
                        help="Override moon center X (auto-loaded from sidecar if omitted)")
    parser.add_argument("--moon-cy",     type=int, default=None,
                        help="Override moon center Y")
    parser.add_argument("--moon-radius", type=int, default=None,
                        help="Override moon radius")
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        events_path=args.events,
        start_sec=args.start,
        end_sec=args.end,
        half_window=args.half_window,
        sigma_k=args.sigma_k,
        min_residual=args.min_residual,
        min_area=args.min_area,
        max_area=args.max_area,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        link_radius=args.link_radius,
        moon_cx=args.moon_cx,
        moon_cy=args.moon_cy,
        moon_radius=args.moon_radius,
    )


if __name__ == "__main__":
    main()
