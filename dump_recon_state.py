"""Dump a reconstruction sidecar `state_dir` to a persistent location with
PNG visualizations for validation.

For each frame in state_dir/phase_a_work/, writes:
    <out>/raw/                    raw .npy copies (rgb / depth_m / c2w)
    <out>/rgb/frame_NNN.png       RGB JPEG-quality PNG
    <out>/depth/frame_NNN.png     depth colormap (turbo, near=blue / far=red)
    <out>/mask_overlay/frame_NNN.png   RGB + 3-color mask overlay
                                  (drawer=green, cabinet door=red, cabinet body=blue)
    <out>/per_part_mask/<part>/frame_NNN.png   per-part isolated mask
    <out>/manifest.txt            human-readable summary
And copies parts.json / joint.json / manifest.json verbatim.

Usage:
    python dump_recon_state.py <state_dir> <out_dir> [--label LABEL]

Example:
    python dump_recon_state.py \
        /tmp/recon_state_40147_r3cara_s \
        experiments/test1_drawer_instr_vflip_buggy/state \
        --label "Test 1 — drawer instruction (PRE v-flip-fix)"
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Color per known SAM3 prompt — used for overlay & per-part
PART_COLORS = {
    "drawer":      (60, 220, 80),    # green
    "cabinet door": (230, 70, 70),   # red
    "cabinet body": (70, 130, 230),  # blue
}
DEFAULT_COLOR = (200, 200, 60)       # yellow fallback for unknown prompts


def colorize_depth(depth_m, near=0.3, far=3.0):
    """Map depth in meters → 8-bit RGB with a turbo-like ramp.
    Invalid (=0 or NaN) pixels rendered as black."""
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    # Simple 5-stop ramp blue→cyan→green→yellow→red (poor-man's turbo, no matplotlib)
    stops = np.array(
        [[12, 12, 168], [50, 200, 220], [60, 220, 80], [240, 230, 50], [220, 50, 50]],
        dtype=np.float32,
    )
    n = stops.shape[0] - 1
    pos = norm * n
    lo = np.clip(np.floor(pos).astype(np.int32), 0, n - 1)
    t = (pos - lo)[..., None]
    rgb = (1.0 - t) * stops[lo] + t * stops[lo + 1]
    rgb = rgb.astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def overlay_mask(rgb, mask_bool, color, alpha=0.45):
    """RGB uint8 (H,W,3) + bool mask (H,W) → blended uint8 (H,W,3)."""
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return out.clip(0, 255).astype(np.uint8)


def dump(state_dir: Path, out_dir: Path, label: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    (out_dir / "rgb").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "mask_overlay").mkdir(exist_ok=True)
    (out_dir / "per_part_mask").mkdir(exist_ok=True)

    # Copy bookkeeping JSONs verbatim — small, useful for validation.
    for fn in ("manifest.json", "parts.json", "joint.json"):
        src = state_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)

    parts_info = {}
    if (state_dir / "parts.json").exists():
        parts_info = json.loads((state_dir / "parts.json").read_text())

    # Discover frames from phase_a_work — names look like "frame_NNN_fNNN_rgb.npy"
    pa = state_dir / "phase_a_work"
    if not pa.exists():
        print(f"  [WARN] no phase_a_work/ in {state_dir}")
        return

    rgb_files = sorted(pa.glob("frame_*_rgb.npy"))
    print(f"  frames: {len(rgb_files)}")

    # Frame_idx → numeric (use the first NNN before _f)
    def _idx(p: Path) -> int:
        m = re.match(r"frame_(\d+)_", p.name)
        return int(m.group(1)) if m else -1

    summary_lines = []
    if label:
        summary_lines.append(f"Label: {label}")
    summary_lines.append(f"State dir: {state_dir}")
    summary_lines.append(f"Frames: {len(rgb_files)}")
    summary_lines.append(f"Parts: {parts_info}")

    for rgb_path in rgb_files:
        i = _idx(rgb_path)
        stem = f"frame_{i:03d}"
        depth_path = pa / rgb_path.name.replace("_rgb.npy", "_depth_m.npy")
        c2w_path = pa / rgb_path.name.replace("_rgb.npy", "_c2w.npy")

        rgb = np.load(rgb_path)
        depth = np.load(depth_path) if depth_path.exists() else None

        # Copy raw arrays
        for src in (rgb_path, depth_path, c2w_path):
            if src.exists():
                shutil.copy2(src, out_dir / "raw" / src.name)

        # RGB
        Image.fromarray(rgb).save(out_dir / "rgb" / f"{stem}.png")

        # Depth colorized
        if depth is not None:
            dvis = colorize_depth(depth)
            Image.fromarray(dvis).save(out_dir / "depth" / f"{stem}.png")

        # Per-part masks + 3-color overlay
        overlay = rgb.copy()
        for part_dir in (state_dir / "per_part").iterdir() if (state_dir / "per_part").exists() else []:
            part = part_dir.name
            color = PART_COLORS.get(part, DEFAULT_COLOR)
            mh = part_dir / "mask_history" / f"{i:06d}.npy"
            if not mh.exists():
                continue
            m = np.load(mh).astype(bool)
            if m.shape != rgb.shape[:2]:
                continue

            # per-part isolated viz
            (out_dir / "per_part_mask" / part).mkdir(parents=True, exist_ok=True)
            iso = overlay_mask(rgb, m, color, alpha=0.55)
            Image.fromarray(iso).save(out_dir / "per_part_mask" / part / f"{stem}.png")

            # cumulative 3-color overlay
            overlay = overlay_mask(overlay, m, color, alpha=0.42)

        Image.fromarray(overlay).save(out_dir / "mask_overlay" / f"{stem}.png")

        if i == _idx(rgb_files[0]):
            summary_lines.append(
                f"Frame 0 RGB shape={rgb.shape} dtype={rgb.dtype} "
                f"depth shape={None if depth is None else depth.shape} "
                f"depth_range=[{None if depth is None else float(depth[depth>0].min()):.3f}, "
                f"{None if depth is None else float(depth.max()):.3f}]m"
            )

    (out_dir / "manifest.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"  → {out_dir}")
    for fn in ("manifest.txt", "parts.json", "joint.json"):
        if (out_dir / fn).exists():
            print(f"      ./{fn}")
    print(f"      ./rgb/ ./depth/ ./mask_overlay/ ./per_part_mask/<part>/ ./raw/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if not args.state_dir.exists():
        print(f"ERROR: state_dir does not exist: {args.state_dir}")
        sys.exit(1)
    dump(args.state_dir, args.out_dir, args.label)


if __name__ == "__main__":
    main()
