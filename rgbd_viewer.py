#!/usr/bin/env python3
"""
Fast RGB-D browser for reconstruction frame dumps.

Expected files in DIR:
    frame_<idx>_<label>_rgb.npy      (H, W, 3)
    frame_<idx>_<label>_depth_m.npy  (H, W), metres, 0 or non-finite = invalid
    frame_<idx>_<label>_c2w.npy      (4, 4), optional camera-to-world matrix

Examples:
    conda run -n owlsam python rgbd_viewer.py recon_output/40147_1778791848/recon_state/phase_a_work
    conda run -n owlsam python rgbd_viewer.py --sheet /tmp/rgbd_sheet.png --no-view

Interactive keys:
    right/space/n  next frame        left/p      previous frame
    ]              +5 frames         [           -5 frames
    home/end       first/last frame  0-9 enter   jump to typed frame number
    m              cycle view mode   r           local/global depth range
    a              autoplay          +/-         autoplay speed
    c              save contact sheet in DIR     s save current view in DIR
    t              cycle part mask    x           export current frame masks
    h              show/hide help    q/escape    quit
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np

_DEFAULT_DIR = Path.home() / (
    "projects/VLM_Planning_for_Articulated_Manipulation/"
    "recon_output/40147_1778791848/recon_state/phase_a_work"
)
_FRAME_RE = re.compile(r"^frame_(\d+)_(.+)_rgb\.npy$")


@dataclass(frozen=True)
class Frame:
    idx: int
    label: str
    rgb_path: Path
    depth_path: Path | None
    c2w_path: Path | None

    @property
    def stem(self) -> str:
        return f"frame_{self.idx:03d}_{self.label}"


@dataclass(frozen=True)
class PartMask:
    name: str
    mask_dir: Path
    color: tuple[int, int, int]

    def path_for(self, frame_idx: int) -> Path:
        return self.mask_dir / f"{frame_idx:06d}.npy"


def discover_frames(work_dir: Path) -> list[Frame]:
    frames: list[Frame] = []
    for path in work_dir.iterdir():
        match = _FRAME_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        label = match.group(2)
        prefix = f"frame_{match.group(1)}_{label}"
        depth_path = work_dir / f"{prefix}_depth_m.npy"
        c2w_path = work_dir / f"{prefix}_c2w.npy"
        frames.append(
            Frame(
                idx=idx,
                label=label,
                rgb_path=path,
                depth_path=depth_path if depth_path.exists() else None,
                c2w_path=c2w_path if c2w_path.exists() else None,
            )
        )
    return sorted(frames, key=lambda f: f.idx)


def default_mask_root(work_dir: Path) -> Path:
    if work_dir.name == "phase_a_work":
        return work_dir.parent / "per_part"
    return work_dir / "per_part"


def discover_part_masks(mask_root: Path) -> list[PartMask]:
    palette = [
        (230, 75, 65),
        (58, 145, 220),
        (75, 170, 105),
        (230, 165, 55),
        (150, 105, 210),
        (70, 190, 190),
    ]
    parts: list[PartMask] = []
    if not mask_root.is_dir():
        return parts
    for idx, part_dir in enumerate(sorted(p for p in mask_root.iterdir() if p.is_dir())):
        mask_dir = part_dir / "mask_history"
        if mask_dir.is_dir() and any(mask_dir.glob("*.npy")):
            parts.append(PartMask(part_dir.name, mask_dir, palette[idx % len(palette)]))
    return parts


@lru_cache(maxsize=16)
def _load_rgb(path: str) -> np.ndarray:
    rgb = np.load(path)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"RGB array must be HxWx3/4: {path} has shape {rgb.shape}")
    return rgb[..., :3]


@lru_cache(maxsize=16)
def _load_depth(path: str) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be HxW: {path} has shape {depth.shape}")
    return depth


@lru_cache(maxsize=16)
def _load_c2w(path: str) -> np.ndarray:
    c2w = np.load(path)
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w array must be 4x4: {path} has shape {c2w.shape}")
    return c2w


@lru_cache(maxsize=64)
def _load_mask(path: str) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"Mask array must be HxW: {path} has shape {mask.shape}")
    return mask.astype(bool)


def valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def depth_limits(depths: Iterable[np.ndarray], percentiles: tuple[float, float]) -> tuple[float, float]:
    vals = []
    for depth in depths:
        valid = valid_depth(depth)
        if valid.any():
            vals.append(depth[valid])
    if not vals:
        return 0.0, 1.0
    joined = np.concatenate(vals)
    lo, hi = np.percentile(joined, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def summarize(frames: list[Frame]) -> str:
    rgb_shape = _load_rgb(str(frames[0].rgb_path)).shape
    depth_count = sum(f.depth_path is not None for f in frames)
    c2w_count = sum(f.c2w_path is not None for f in frames)
    return (
        f"{len(frames)} frame(s), RGB {rgb_shape}, "
        f"depth {depth_count}/{len(frames)}, c2w {c2w_count}/{len(frames)}"
    )


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def overlay_masks(rgb: np.ndarray, masks: dict[str, np.ndarray], parts: list[PartMask], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    for part in parts:
        mask = masks.get(part.name)
        if mask is None or not mask.any():
            continue
        color = np.array(part.color, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def masked_rgba(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return rgba


def crop_to_mask(array: np.ndarray, mask: np.ndarray, fill_value=0) -> np.ndarray:
    bbox = mask_bbox(mask)
    if bbox is None:
        return array[:0, :0].copy()
    x0, y0, x1, y1 = bbox
    cropped = array[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1]
    cropped[~cropped_mask] = fill_value
    return cropped


class RgbdViewer:
    def __init__(
        self,
        frames: list[Frame],
        work_dir: Path,
        depth_percentiles: tuple[float, float],
        parts: list[PartMask] | None = None,
        start: int = 0,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.frames = frames
        self.work_dir = work_dir
        self.i = max(0, min(start, len(frames) - 1))
        self.mode = "split"
        self.use_global_range = False
        self.depth_percentiles = depth_percentiles
        self.parts = parts or []
        self.part_i = 0
        self.global_range: tuple[float, float] | None = None
        self.jump_buffer = ""
        self.help_visible = True
        self.playing = False
        self.interval_ms = 350

        self.fig, self.axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._tick)
        if getattr(self.fig.canvas, "manager", None):
            self.fig.canvas.manager.set_window_title("RGB-D viewer")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def run(self) -> None:
        self.render()
        self.plt.show()

    def frame(self) -> Frame:
        return self.frames[self.i]

    def depth_range(self, depth: np.ndarray) -> tuple[float, float]:
        if not self.use_global_range:
            return depth_limits([depth], self.depth_percentiles)
        if self.global_range is None:
            print("[rgbd_viewer] computing global depth range...")
            self.global_range = depth_limits(
                (_load_depth(str(f.depth_path)) for f in self.frames if f.depth_path),
                self.depth_percentiles,
            )
        return self.global_range

    def frame_masks(self, frame: Frame) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for part in self.parts:
            path = part.path_for(frame.idx)
            if path.exists():
                masks[part.name] = _load_mask(str(path))
        return masks

    def selected_part(self) -> PartMask | None:
        if not self.parts:
            return None
        return self.parts[self.part_i % len(self.parts)]

    def render(self) -> None:
        frame = self.frame()
        rgb = _load_rgb(str(frame.rgb_path))
        depth = _load_depth(str(frame.depth_path)) if frame.depth_path else None
        masks = self.frame_masks(frame) if self.parts else {}

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        if self.mode == "rgb":
            self.axes[0].imshow(rgb)
            self.axes[0].set_title("RGB")
            self.axes[1].text(0.5, 0.5, self.info_text(frame, depth), ha="center", va="center")
            self.axes[1].set_title("Frame info")
        elif self.mode == "depth":
            self.show_depth(self.axes[0], depth)
            self.axes[1].text(0.5, 0.5, self.info_text(frame, depth), ha="center", va="center")
            self.axes[1].set_title("Frame info")
        elif self.mode == "overlay":
            self.show_overlay(self.axes[0], rgb, depth)
            self.axes[1].imshow(rgb)
            self.axes[1].set_title("RGB reference")
        elif self.mode == "mask":
            self.show_all_masks(self.axes[0], rgb, masks)
            self.axes[1].text(0.02, 0.98, self.mask_info_text(frame, masks), ha="left", va="top", family="monospace")
            self.axes[1].set_title("Mask stats")
        elif self.mode == "part":
            self.show_selected_part(self.axes[0], rgb, masks)
            self.show_selected_part_crop(self.axes[1], rgb, depth, masks)
        else:
            self.axes[0].imshow(rgb)
            self.axes[0].set_title("RGB")
            self.show_depth(self.axes[1], depth)

        title = (
            f"{frame.stem}  [{self.i + 1}/{len(self.frames)}]  "
            f"mode={self.mode}  range={'global' if self.use_global_range else 'local'}"
        )
        if self.playing:
            title += f"  autoplay={self.interval_ms}ms"
        if self.jump_buffer:
            title += f"  jump:{self.jump_buffer}"
        self.fig.suptitle(title, fontsize=11)
        if self.help_visible:
            self.fig.text(
                0.5,
                0.015,
                "arrows/space navigate | m mode | r range | a autoplay | digits+enter jump | "
                "c contact sheet | s snapshot | t part | x export frame masks | h help | q quit",
                ha="center",
                fontsize=9,
            )
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        self.fig.canvas.draw_idle()

    def show_depth(self, ax, depth: np.ndarray | None) -> None:
        if depth is None:
            ax.text(0.5, 0.5, "depth missing", ha="center", va="center")
            ax.set_title("Depth")
            return
        valid = valid_depth(depth)
        lo, hi = self.depth_range(depth)
        disp = np.ma.masked_where(~valid, depth)
        ax.imshow(disp, cmap="turbo", vmin=lo, vmax=hi)
        if valid.any():
            ax.set_title(f"Depth {lo:.3f}-{hi:.3f} m, valid {valid.mean() * 100:.1f}%")
        else:
            ax.set_title("Depth: no valid pixels")

    def show_overlay(self, ax, rgb: np.ndarray, depth: np.ndarray | None) -> None:
        if depth is None:
            ax.imshow(rgb)
            ax.set_title("Overlay: depth missing")
            return
        import matplotlib.pyplot as plt

        valid = valid_depth(depth)
        lo, hi = self.depth_range(depth)
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        color = (plt.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[valid] = (0.55 * rgb[valid] + 0.45 * color[valid]).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title("RGB + depth overlay")

    def show_all_masks(self, ax, rgb: np.ndarray, masks: dict[str, np.ndarray]) -> None:
        if not masks:
            ax.imshow(rgb)
            ax.set_title("No masks found")
            return
        ax.imshow(overlay_masks(rgb, masks, self.parts))
        for part in self.parts:
            mask = masks.get(part.name)
            if mask is None:
                continue
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            color = np.array(part.color) / 255.0
            ax.add_patch(self.plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=1.5))
            ax.text(x0, max(0, y0 - 3), part.name, color=color, fontsize=9, weight="bold")
        ax.set_title("Part masks overlay")

    def show_selected_part(self, ax, rgb: np.ndarray, masks: dict[str, np.ndarray]) -> None:
        part = self.selected_part()
        if part is None or part.name not in masks:
            ax.imshow(rgb)
            ax.set_title("No selected part mask")
            return
        mask = masks[part.name]
        ax.imshow(overlay_masks(rgb, {part.name: mask}, [part], alpha=0.6))
        bbox = mask_bbox(mask)
        if bbox:
            x0, y0, x1, y1 = bbox
            color = np.array(part.color) / 255.0
            ax.add_patch(self.plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=2.0))
        ax.set_title(f"Selected part: {part.name}")

    def show_selected_part_crop(self, ax, rgb: np.ndarray, depth: np.ndarray | None, masks: dict[str, np.ndarray]) -> None:
        part = self.selected_part()
        if part is None or part.name not in masks:
            ax.text(0.5, 0.5, "part mask missing", ha="center", va="center")
            ax.set_title("Extracted part")
            return
        mask = masks[part.name]
        bbox = mask_bbox(mask)
        if bbox is None:
            ax.text(0.5, 0.5, "empty mask", ha="center", va="center")
            ax.set_title(f"{part.name}: empty")
            return
        crop = crop_to_mask(rgb, mask, fill_value=0)
        ax.imshow(crop)
        x0, y0, x1, y1 = bbox
        title = f"{part.name} crop {x1 - x0}x{y1 - y0}, pixels={int(mask.sum())}"
        if depth is not None:
            dmask = mask & valid_depth(depth)
            if dmask.any():
                title += f", depth={depth[dmask].min():.2f}-{depth[dmask].max():.2f}m"
        ax.set_title(title)

    def mask_info_text(self, frame: Frame, masks: dict[str, np.ndarray]) -> str:
        lines = [f"frame {frame.idx:06d}", "part              pixels   coverage   bbox"]
        union = None
        overlap = None
        for part in self.parts:
            mask = masks.get(part.name)
            if mask is None:
                lines.append(f"{part.name:<16} missing")
                continue
            if union is None:
                union = np.zeros(mask.shape, dtype=bool)
                overlap = np.zeros(mask.shape, dtype=bool)
            overlap |= union & mask
            union |= mask
            bbox = mask_bbox(mask)
            bbox_text = "-" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            lines.append(f"{part.name:<16} {int(mask.sum()):>7}  {mask.mean() * 100:>7.2f}%  {bbox_text}")
        if union is not None and overlap is not None:
            lines.append("")
            lines.append(f"union pixels:    {int(union.sum())} ({union.mean() * 100:.2f}%)")
            lines.append(f"overlap pixels:  {int(overlap.sum())}")
        return "\n".join(lines)

    def info_text(self, frame: Frame, depth: np.ndarray | None) -> str:
        lines = [f"idx: {frame.idx}", f"label: {frame.label}", f"rgb: {frame.rgb_path.name}"]
        if depth is not None:
            valid = valid_depth(depth)
            lines.append(f"depth: {frame.depth_path.name if frame.depth_path else 'missing'}")
            if valid.any():
                lines.append(f"depth min/max: {depth[valid].min():.4f} / {depth[valid].max():.4f} m")
                lines.append(f"valid pixels: {valid.mean() * 100:.2f}%")
            else:
                lines.append("valid pixels: 0%")
        if frame.c2w_path:
            c2w = _load_c2w(str(frame.c2w_path))
            xyz = c2w[:3, 3]
            lines.append(f"cam xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
        if self.parts:
            masks = self.frame_masks(frame)
            lines.append(f"parts: {', '.join(p.name for p in self.parts)}")
            for part in self.parts:
                mask = masks.get(part.name)
                if mask is not None:
                    lines.append(f"  {part.name}: {int(mask.sum())} px")
        return "\n".join(lines)

    def step(self, delta: int) -> None:
        self.i = (self.i + delta) % len(self.frames)
        self.render()

    def on_key(self, event) -> None:
        key = event.key or ""
        if key in ("q", "escape"):
            self.timer.stop()
            self.plt.close(self.fig)
        elif key in ("right", "down", " ", "n"):
            self.step(1)
        elif key in ("left", "up", "p"):
            self.step(-1)
        elif key == "]":
            self.step(5)
        elif key == "[":
            self.step(-5)
        elif key == "home":
            self.i = 0
            self.render()
        elif key == "end":
            self.i = len(self.frames) - 1
            self.render()
        elif key == "m":
            modes = ["split", "rgb", "depth", "overlay"]
            if self.parts:
                modes += ["mask", "part"]
            self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
            self.render()
        elif key == "t":
            if self.parts:
                self.part_i = (self.part_i + 1) % len(self.parts)
                if self.mode not in ("part", "mask"):
                    self.mode = "part"
                self.render()
        elif key == "r":
            self.use_global_range = not self.use_global_range
            self.render()
        elif key == "a":
            self.toggle_autoplay()
        elif key in ("+", "="):
            self.interval_ms = max(50, int(self.interval_ms * 0.75))
            self.restart_timer()
            self.render()
        elif key in ("-", "_"):
            self.interval_ms = min(3000, int(self.interval_ms * 1.25))
            self.restart_timer()
            self.render()
        elif key == "h":
            self.help_visible = not self.help_visible
            self.render()
        elif key == "s":
            self.save_snapshot()
        elif key == "c":
            out = self.work_dir / "rgbd_contact_sheet.png"
            save_contact_sheet(self.frames, out, self.depth_percentiles)
            print(f"[rgbd_viewer] saved {out}")
        elif key == "x":
            if self.parts:
                out = self.work_dir / "mask_extract_current"
                export_mask_extractions([self.frame()], self.parts, out)
                print(f"[rgbd_viewer] exported current frame masks to {out}")
        elif key.isdigit():
            self.jump_buffer += key
            self.render()
        elif key in ("enter", "return") and self.jump_buffer:
            target = int(self.jump_buffer)
            self.jump_buffer = ""
            by_idx = {f.idx: n for n, f in enumerate(self.frames)}
            self.i = by_idx.get(target, max(0, min(target, len(self.frames) - 1)))
            self.render()
        elif key == "backspace":
            self.jump_buffer = self.jump_buffer[:-1]
            self.render()

    def toggle_autoplay(self) -> None:
        self.playing = not self.playing
        self.restart_timer()
        self.render()

    def restart_timer(self) -> None:
        self.timer.stop()
        self.timer.interval = self.interval_ms
        if self.playing:
            self.timer.start()

    def _tick(self) -> bool:
        if self.playing:
            self.step(1)
        return True

    def save_snapshot(self) -> None:
        frame = self.frame()
        out = self.work_dir / f"{frame.stem}_{self.mode}.png"
        self.fig.savefig(out, dpi=160)
        print(f"[rgbd_viewer] saved {out}")


def save_contact_sheet(
    frames: list[Frame],
    out_path: Path,
    depth_percentiles: tuple[float, float],
    cols: int = 5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
    global_range = depth_limits(
        (_load_depth(str(f.depth_path)) for f in frames if f.depth_path),
        depth_percentiles,
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, frame in zip(axes.flat, frames):
        rgb = _load_rgb(str(frame.rgb_path))
        ax.imshow(rgb)
        title = f"{frame.idx:03d}"
        if frame.depth_path:
            depth = _load_depth(str(frame.depth_path))
            valid = valid_depth(depth)
            if valid.any():
                title += f"  d:{depth[valid].min():.2f}-{depth[valid].max():.2f}m"
            lo, hi = global_range
            norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            color = (plt.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
            overlay = rgb.copy()
            overlay[valid] = (0.62 * rgb[valid] + 0.38 * color[valid]).astype(np.uint8)
            ax.imshow(overlay)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_mask_sheet(frames: list[Frame], parts: list[PartMask], out_path: Path, cols: int = 5) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, frame in zip(axes.flat, frames):
        rgb = _load_rgb(str(frame.rgb_path))
        masks = {}
        for part in parts:
            path = part.path_for(frame.idx)
            if path.exists():
                masks[part.name] = _load_mask(str(path))
        ax.imshow(overlay_masks(rgb, masks, parts))
        ax.set_title(f"{frame.idx:03d}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def export_mask_extractions(frames: list[Frame], parts: list[PartMask], out_dir: Path) -> None:
    from PIL import Image

    out_dir = out_dir.expanduser().resolve()
    rows = []
    for frame in frames:
        rgb = _load_rgb(str(frame.rgb_path))
        depth = _load_depth(str(frame.depth_path)) if frame.depth_path else None
        for part in parts:
            mask_path = part.path_for(frame.idx)
            if not mask_path.exists():
                continue
            mask = _load_mask(str(mask_path))
            bbox = mask_bbox(mask)
            part_dir = out_dir / part.name
            part_dir.mkdir(parents=True, exist_ok=True)

            stem = f"{frame.idx:06d}"
            Image.fromarray(mask.astype(np.uint8) * 255).save(part_dir / f"{stem}_mask.png")
            Image.fromarray(masked_rgba(rgb, mask)).save(part_dir / f"{stem}_rgba.png")

            x0 = y0 = x1 = y1 = ""
            dmin = dmax = dmean = ""
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                rgb_crop = crop_to_mask(rgb, mask, fill_value=0)
                rgba_crop = masked_rgba(rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1])
                Image.fromarray(rgb_crop).save(part_dir / f"{stem}_rgb_crop.png")
                Image.fromarray(rgba_crop).save(part_dir / f"{stem}_rgba_crop.png")
                if depth is not None:
                    depth_crop = crop_to_mask(depth, mask, fill_value=np.nan)
                    np.save(part_dir / f"{stem}_depth_crop_m.npy", depth_crop)
                    valid = mask & valid_depth(depth)
                    if valid.any():
                        dmin = float(depth[valid].min())
                        dmax = float(depth[valid].max())
                        dmean = float(depth[valid].mean())

            rows.append(
                {
                    "frame_idx": frame.idx,
                    "frame_label": frame.label,
                    "part": part.name,
                    "mask_pixels": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "bbox_x0": x0,
                    "bbox_y0": y0,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "depth_min_m": dmin,
                    "depth_max_m": dmax,
                    "depth_mean_m": dmean,
                    "mask_path": str(mask_path),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mask_stats.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "frame_idx",
            "frame_label",
            "part",
            "mask_pixels",
            "coverage",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "depth_min_m",
            "depth_max_m",
            "depth_mean_m",
            "mask_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse frame_* RGB-D .npy dumps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dir", nargs="?", default=str(_DEFAULT_DIR), help="phase_a_work directory")
    parser.add_argument("--start", type=int, default=0, help="start at list position or frame idx")
    parser.add_argument("--depth-percentiles", nargs=2, type=float, default=(2.0, 98.0))
    parser.add_argument("--mask-root", type=Path, help="per_part directory containing <part>/mask_history")
    parser.add_argument("--sheet", type=Path, help="write a contact sheet PNG")
    parser.add_argument("--mask-sheet", type=Path, help="write a mask overlay contact sheet PNG")
    parser.add_argument("--export-masks", type=Path, help="export per-frame per-part masks/crops/depth crops")
    parser.add_argument("--no-view", action="store_true", help="do not open the interactive viewer")
    return parser.parse_args()


def ensure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path.home() / ".config" / "matplotlib"
    if config_dir.exists() and os.access(config_dir, os.W_OK):
        return
    cache_dir = Path("/tmp") / f"rgbd_viewer_mpl_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def main() -> int:
    args = parse_args()
    ensure_matplotlib_cache()
    work_dir = Path(os.path.expanduser(args.dir)).resolve()
    if not work_dir.is_dir():
        print(f"[rgbd_viewer] not a directory: {work_dir}", file=sys.stderr)
        return 2

    frames = discover_frames(work_dir)
    if not frames:
        print(f"[rgbd_viewer] no frame_*_rgb.npy found in {work_dir}", file=sys.stderr)
        return 2

    print(f"[rgbd_viewer] {work_dir}")
    print(f"[rgbd_viewer] {summarize(frames)}")
    mask_root = args.mask_root.expanduser().resolve() if args.mask_root else default_mask_root(work_dir)
    parts = discover_part_masks(mask_root)
    if parts:
        print(f"[rgbd_viewer] masks: {mask_root}")
        print(f"[rgbd_viewer] parts: {', '.join(p.name for p in parts)}")
    else:
        print(f"[rgbd_viewer] masks: none found at {mask_root}")

    if args.sheet:
        save_contact_sheet(frames, args.sheet.expanduser().resolve(), tuple(args.depth_percentiles))
        print(f"[rgbd_viewer] saved {args.sheet.expanduser().resolve()}")
    if args.mask_sheet:
        if not parts:
            print("[rgbd_viewer] --mask-sheet requested but no part masks found", file=sys.stderr)
            return 2
        save_mask_sheet(frames, parts, args.mask_sheet.expanduser().resolve())
        print(f"[rgbd_viewer] saved {args.mask_sheet.expanduser().resolve()}")
    if args.export_masks:
        if not parts:
            print("[rgbd_viewer] --export-masks requested but no part masks found", file=sys.stderr)
            return 2
        export_mask_extractions(frames, parts, args.export_masks.expanduser().resolve())
        print(f"[rgbd_viewer] exported masks to {args.export_masks.expanduser().resolve()}")

    if args.no_view:
        return 0

    by_idx = {f.idx: i for i, f in enumerate(frames)}
    start = by_idx.get(args.start, args.start)
    viewer = RgbdViewer(frames, work_dir, tuple(args.depth_percentiles), parts=parts, start=start)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
