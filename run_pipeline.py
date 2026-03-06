"""
Full TLP detection pipeline.

Usage:
    python run_pipeline.py input.mp4 --start 2 --end 10
    python run_pipeline.py input.mp4 --start 2 --end 10 --min-residual 20
"""

import argparse
import sys
from register_frames import process_video as register
from detect_flashes import process as detect, DEFAULT_HALF_WINDOW, DEFAULT_SIGMA_K, \
    DEFAULT_MIN_RESIDUAL, DEFAULT_MIN_AREA, DEFAULT_MAX_AREA, \
    DEFAULT_MIN_FRAMES, DEFAULT_MAX_FRAMES, DEFAULT_LINK_RADIUS


def main():
    parser = argparse.ArgumentParser(description="TLP detector — full pipeline")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--start",        type=float, default=0.0,  help="Start time in seconds")
    parser.add_argument("--end",          type=float, default=-1.0, help="End time in seconds (-1 = full video)")
    parser.add_argument("--registered",   default="registered.mp4",  help="Intermediate registered video")
    parser.add_argument("--output",       default="detections.mp4",  help="Annotated output video")
    parser.add_argument("--events",       default="events.json",      help="Events JSON")
    # Detection tuning
    parser.add_argument("--half-window",  type=int,   default=DEFAULT_HALF_WINDOW)
    parser.add_argument("--sigma-k",      type=float, default=DEFAULT_SIGMA_K)
    parser.add_argument("--min-residual", type=float, default=DEFAULT_MIN_RESIDUAL,
                        help="Hard minimum residual in DN (default 30, lower = more sensitive)")
    parser.add_argument("--min-area",     type=int,   default=DEFAULT_MIN_AREA)
    parser.add_argument("--max-area",     type=int,   default=DEFAULT_MAX_AREA)
    parser.add_argument("--min-frames",   type=int,   default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--max-frames",   type=int,   default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--link-radius",  type=float, default=DEFAULT_LINK_RADIUS)
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1 — Registration")
    print("=" * 60)
    register(args.input, args.registered, args.start, args.end)

    print("\n" + "=" * 60)
    print("STEP 2 — Flash detection")
    print("=" * 60)
    events = detect(
        input_path=args.registered,
        output_path=args.output,
        events_path=args.events,
        start_sec=0.0,
        end_sec=-1.0,
        half_window=args.half_window,
        sigma_k=args.sigma_k,
        min_residual=args.min_residual,
        min_area=args.min_area,
        max_area=args.max_area,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        link_radius=args.link_radius,
    )

    print(f"\nDone. {len(events)} candidate event(s) found.")
    print(f"  Registered video : {args.registered}")
    print(f"  Detections video : {args.output}")
    print(f"  Events JSON      : {args.events}")


if __name__ == "__main__":
    main()
