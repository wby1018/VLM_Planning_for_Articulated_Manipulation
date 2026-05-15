"""Live visualizer for the reconstruction sidecar's state_dir.

Reads sidecar's on-disk dumps (phase_a_work/ + per_part/) and renders four
panes in a single Tk window at ~5 Hz:

  ┌──────────────────────┬──────────────────────┐
  │  RGB (latest frame)  │  Depth (turbo)       │
  ├──────────────────────┼──────────────────────┤
  │  Per-part mask       │  Accumulated 3D pts  │
  │  overlay (RGB +      │  per part (drawer=G, │
  │  drawer=G, door=R,   │  door=R, body=B)     │
  │  body=B)             │                      │
  └──────────────────────┴──────────────────────┘
                  Frame N | status

Design choices (per user 2026-05-13 evaluation):
  - Independent process, polling at fixed interval — no coupling with the
    SAPIEN main loop, so no risk of stalling the simulator's render thread.
  - Matplotlib + TkAgg, single figure with 2×2 subplots (last pane is 3D).
    matplotlib 3D is slower than Open3D's Visualizer but embeds cleanly in
    Tk and avoids a second window.
  - Reads the same files dump_recon_state.py reads — same color convention
    (PART_COLORS) for visual continuity with the offline tool.

Usage:
    python live_visualizer.py [--state-dir PATH] [--interval-ms N] \
                              [--max-points 3000] [--depth-max-m 5.0]

If --state-dir is omitted, auto-picks the most recently mtime'd
`/tmp/recon_state_*` directory and re-checks on each refresh so a freshly
started sidecar's state_dir gets picked up automatically.
"""
from __future__ import annotations

import argparse
import re
import time
import tkinter as tk
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import get_cmap
import numpy as np

# Match dump_recon_state.py's color convention so RGB-overlay reads the same.
# Tuple is (R, G, B) uint8.
PART_COLORS = {
    "drawer":       np.array([ 80, 220,  80], dtype=np.uint8),  # green
    "cabinet door": np.array([220,  60,  60], dtype=np.uint8),  # red
    "cabinet_door": np.array([220,  60,  60], dtype=np.uint8),  # filesystem-safe alias
    "cabinet body": np.array([ 60, 100, 220], dtype=np.uint8),  # blue
    "cabinet_body": np.array([ 60, 100, 220], dtype=np.uint8),
}
DEFAULT_COLOR = np.array([200, 200, 60], dtype=np.uint8)

# matplotlib colors (0-1 floats) for 3-D scatter
PART_COLORS_F = {k: v / 255.0 for k, v in PART_COLORS.items()}


_FRAME_RGB_RE = re.compile(r"frame_(\d+)_f(\d+)_rgb\.npy$")


def find_latest_state_dir(prefix: str = "/tmp/recon_state_") -> Optional[Path]:
    """Return the most-recently-modified /tmp/recon_state_* directory, or None."""
    candidates = sorted(
        (p for p in Path("/tmp").glob("recon_state_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_frame_index(state_dir: Path) -> Optional[int]:
    """Largest frame index for which `phase_a_work/frame_NNN_fNNN_rgb.npy` exists."""
    pa = state_dir / "phase_a_work"
    if not pa.is_dir():
        return None
    best = -1
    for p in pa.iterdir():
        m = _FRAME_RGB_RE.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best if best >= 0 else None


def frame_artifacts(state_dir: Path, idx: int):
    """Resolve (rgb_npy, depth_npy) paths for `idx`. May return (None, None) if
    the file got rotated out between latest_frame_index() and us."""
    pa = state_dir / "phase_a_work"
    rgb = next(pa.glob(f"frame_{idx:03d}_f*_rgb.npy"), None)
    depth = next(pa.glob(f"frame_{idx:03d}_f*_depth_m.npy"), None)
    return rgb, depth


def colorize_depth(depth_m: np.ndarray, vmax: float = 5.0) -> np.ndarray:
    """Turbo colormap on a depth array. vmax clips the far end."""
    finite = np.isfinite(depth_m) & (depth_m > 0)
    z = np.where(finite, np.clip(depth_m, 0.0, vmax) / max(vmax, 1e-6), 0.0)
    rgb = (get_cmap("turbo")(z)[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = 0   # paint invalid pixels black
    return rgb


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.45):
    out = rgb.copy()
    color_arr = color.astype(np.float32)
    out[mask] = ((1 - alpha) * out[mask].astype(np.float32) + alpha * color_arr).astype(np.uint8)
    return out


def cumulative_overlay(rgb: np.ndarray, state_dir: Path, idx: int):
    """RGB with all part masks blended in their PART_COLORS."""
    out = rgb.copy()
    per_part = state_dir / "per_part"
    if not per_part.is_dir():
        return out
    for part_dir in sorted(per_part.iterdir()):
        if not part_dir.is_dir():
            continue
        part = part_dir.name
        color = PART_COLORS.get(part, PART_COLORS.get(part.replace("_", " "), DEFAULT_COLOR))
        mh = part_dir / "mask_history" / f"{idx:06d}.npy"
        if not mh.exists():
            continue
        try:
            m = np.load(mh)
        except Exception:
            continue
        if m.shape != out.shape[:2]:
            continue
        m = m > 0
        out = overlay_mask(out, m, color, alpha=0.45)
    return out


def load_part_pts(state_dir: Path, max_pts: int = 3000):
    """Returns {part: (Nx3 array, color_f)} of accumulated depth_pts in world frame."""
    out = {}
    per_part = state_dir / "per_part"
    if not per_part.is_dir():
        return out
    for part_dir in sorted(per_part.iterdir()):
        if not part_dir.is_dir():
            continue
        part = part_dir.name
        f = part_dir / "depth_pts.npy"
        if not f.exists():
            continue
        try:
            pts = np.load(f)
        except Exception:
            continue
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            continue
        pts = pts[:, :3]
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        color = PART_COLORS_F.get(part, PART_COLORS_F.get(part.replace("_", " "), DEFAULT_COLOR / 255.0))
        out[part] = (pts, color)
    return out


class LiveViewer:
    def __init__(self, explicit_state_dir: Optional[Path], interval_ms: int,
                 max_points: int, depth_max_m: float):
        self.explicit_state_dir = explicit_state_dir
        self.interval_ms = interval_ms
        self.max_points = max_points
        self.depth_max_m = depth_max_m
        self.last_idx = -1
        self.last_state_dir: Optional[Path] = None

        self.root = tk.Tk()
        self.root.title("Reconstruction sidecar — live visualizer")
        self.root.geometry("1280x900")

        self.fig = plt.Figure(figsize=(12.4, 8.6), tight_layout=True)
        self.ax_rgb = self.fig.add_subplot(2, 2, 1)
        self.ax_depth = self.fig.add_subplot(2, 2, 2)
        self.ax_overlay = self.fig.add_subplot(2, 2, 3)
        self.ax_3d = self.fig.add_subplot(2, 2, 4, projection="3d")
        for ax, title in [(self.ax_rgb, "RGB"), (self.ax_depth, f"Depth (turbo, 0..{depth_max_m:.1f} m)"),
                          (self.ax_overlay, "Mask overlay (drawer=G, door=R, body=B)"),
                          (self.ax_3d, "Accumulated points per part (world)")]:
            ax.set_title(title, fontsize=10)
            if ax is not self.ax_3d:
                ax.set_xticks([]); ax.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="(waiting for state_dir)")
        tk.Label(self.root, textvariable=self.status_var,
                 anchor="w", font=("monospace", 10)).pack(fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._refresh)

    def _on_close(self):
        self.root.quit()
        self.root.destroy()

    def _pick_state_dir(self) -> Optional[Path]:
        if self.explicit_state_dir is not None:
            return self.explicit_state_dir if self.explicit_state_dir.is_dir() else None
        return find_latest_state_dir()

    def _refresh(self):
        state_dir = self._pick_state_dir()
        if state_dir is None:
            self.status_var.set("(no /tmp/recon_state_* dir found; sidecar not started?)")
            self.root.after(self.interval_ms, self._refresh)
            return

        if state_dir != self.last_state_dir:
            self.last_state_dir = state_dir
            self.last_idx = -1   # force redraw

        idx = latest_frame_index(state_dir)
        if idx is None:
            self.status_var.set(f"{state_dir.name} — no frames yet")
            self.root.after(self.interval_ms, self._refresh)
            return

        if idx == self.last_idx:
            self.root.after(self.interval_ms, self._refresh)
            return

        rgb_p, depth_p = frame_artifacts(state_dir, idx)
        if rgb_p is None or not rgb_p.exists():
            self.root.after(self.interval_ms, self._refresh)
            return
        try:
            rgb = np.load(rgb_p)
            depth = np.load(depth_p) if depth_p and depth_p.exists() else None
        except Exception as e:
            self.status_var.set(f"load error: {e}")
            self.root.after(self.interval_ms, self._refresh)
            return

        # ── RGB
        self.ax_rgb.clear(); self.ax_rgb.set_title("RGB", fontsize=10)
        self.ax_rgb.imshow(rgb); self.ax_rgb.set_xticks([]); self.ax_rgb.set_yticks([])

        # ── Depth turbo
        self.ax_depth.clear()
        self.ax_depth.set_title(f"Depth (turbo, 0..{self.depth_max_m:.1f} m)", fontsize=10)
        if depth is not None:
            self.ax_depth.imshow(colorize_depth(depth, vmax=self.depth_max_m))
        self.ax_depth.set_xticks([]); self.ax_depth.set_yticks([])

        # ── Mask overlay
        self.ax_overlay.clear()
        self.ax_overlay.set_title("Mask overlay (drawer=G, door=R, body=B)", fontsize=10)
        self.ax_overlay.imshow(cumulative_overlay(rgb, state_dir, idx))
        self.ax_overlay.set_xticks([]); self.ax_overlay.set_yticks([])

        # ── 3-D accumulated point clouds
        self.ax_3d.clear()
        self.ax_3d.set_title("Accumulated points per part (world)", fontsize=10)
        parts_pts = load_part_pts(state_dir, max_pts=self.max_points)
        for part, (pts, color) in parts_pts.items():
            self.ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                               s=2, c=[color], label=part, alpha=0.6)
        if parts_pts:
            self.ax_3d.legend(fontsize=8, loc="upper right")
        self.ax_3d.set_xlabel("x"); self.ax_3d.set_ylabel("y"); self.ax_3d.set_zlabel("z")

        self.canvas.draw_idle()
        self.last_idx = idx

        parts_summary = ", ".join(f"{p}:{len(pts)}px3D"
                                  for p, (pts, _) in parts_pts.items()) or "(no part dirs)"
        self.status_var.set(f"{state_dir.name} | frame {idx} | {parts_summary} | "
                            f"refresh {self.interval_ms}ms | {time.strftime('%H:%M:%S')}")

        self.root.after(self.interval_ms, self._refresh)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", type=Path, default=None,
                    help="Explicit sidecar state_dir to watch. Default: auto-pick latest /tmp/recon_state_*")
    ap.add_argument("--interval-ms", type=int, default=200,
                    help="Refresh interval in ms (default 200 = 5 Hz)")
    ap.add_argument("--max-points", type=int, default=3000,
                    help="Subsample per-part 3-D point cloud for performance (default 3000)")
    ap.add_argument("--depth-max-m", type=float, default=5.0,
                    help="Far-plane clip for depth colormap (default 5 m)")
    args = ap.parse_args()

    viewer = LiveViewer(
        explicit_state_dir=args.state_dir,
        interval_ms=args.interval_ms,
        max_points=args.max_points,
        depth_max_m=args.depth_max_m,
    )
    print(f"[live_visualizer] watching {'(auto)' if args.state_dir is None else args.state_dir}, "
          f"{args.interval_ms}ms interval")
    viewer.run()


if __name__ == "__main__":
    main()
