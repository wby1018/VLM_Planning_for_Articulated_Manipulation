#!/usr/bin/env python3
"""
Stable Matplotlib URDF motion previewer.

This is a conservative fallback for environments where Open3D's GUI window is
unstable. It shows each visual mesh as a downsampled point cloud, with sliders
and numeric inputs for movable joints.

Examples:
    conda run -n owlsam python urdf_motion_viewer_mpl.py
    conda run -n owlsam python urdf_motion_viewer_mpl.py --open 1.0
    conda run -n owlsam python urdf_motion_viewer_mpl.py --q cabinet_door_joint=0.4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

import urdf_viewer as uv


def movable_joints(robot: uv.Robot) -> list[uv.Joint]:
    return [j for j in robot.joints if j.joint_type != "fixed"]


def joint_limits(joint: uv.Joint) -> tuple[float, float]:
    if joint.lower is not None and joint.upper is not None and joint.lower < joint.upper:
        return joint.lower, joint.upper
    if joint.joint_type in ("revolute", "continuous"):
        return -np.pi, np.pi
    if joint.joint_type == "prismatic":
        return -1.0, 1.0
    return 0.0, 1.0


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.c_[points, np.ones(len(points))]
    return (homo @ transform.T)[:, :3]


def load_link_points(robot: uv.Robot, max_points_per_visual: int) -> list[dict]:
    rng = np.random.default_rng(7)
    items = []
    for link_idx, (link_name, link) in enumerate(robot.links.items()):
        for visual in link.visuals:
            mesh = uv.load_visual_mesh(visual.mesh_path, visual.scale)
            vertices = np.asarray(mesh.vertices, dtype=float)
            if len(vertices) > max_points_per_visual:
                choice = rng.choice(len(vertices), size=max_points_per_visual, replace=False)
                vertices = vertices[choice]
            base = transform_points(vertices, visual.transform)
            color = uv.color_for_link(link_idx)[:3] / 255.0
            items.append({"link": link_name, "points": base, "color": color})
    return items


def equal_axes(ax, points: np.ndarray) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float((hi - lo).max()) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


class MatplotlibMotionViewer:
    def __init__(self, robot: uv.Robot, q: dict[str, float], max_points_per_visual: int):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.Button = Button
        self.Slider = Slider
        self.TextBox = TextBox
        self.robot = robot
        self.q = dict(q)
        self.joints = movable_joints(robot)
        self.items = load_link_points(robot, max_points_per_visual)
        self.scatters = []
        self.axis_lines = {}
        self.textboxes = {}
        self.sliders = {}
        self.buttons = []
        self._syncing = False

        height = max(7.0, 6.0 + 0.7 * len(self.joints))
        self.fig = plt.figure(figsize=(11.5, height))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.18 + 0.065 * len(self.joints))
        if getattr(self.fig.canvas, "manager", None):
            self.fig.canvas.manager.set_window_title("URDF Motion Viewer (Matplotlib)")

        for item in self.items:
            sc = self.ax.scatter([], [], [], s=1.0, c=[item["color"]], depthshade=False)
            self.scatters.append(sc)

        self._setup_axes()
        self._setup_controls()
        self._apply_pose(reset_limits=True)

    def _setup_axes(self) -> None:
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_title(f"{self.robot.name}    drag to rotate, wheel to zoom")

    def _setup_controls(self) -> None:
        if not self.joints:
            return

        row_h = 0.045
        gap = 0.018
        y = 0.04 + (len(self.joints) - 1) * (row_h + gap)
        for joint in self.joints:
            lo, hi = joint_limits(joint)
            slider_ax = self.fig.add_axes([0.18, y, 0.54, row_h])
            text_ax = self.fig.add_axes([0.75, y, 0.15, row_h])
            slider = self.Slider(slider_ax, joint.name, lo, hi, valinit=self.q.get(joint.name, lo), valfmt="%.5f")
            textbox = self.TextBox(text_ax, "", initial=f"{self.q.get(joint.name, lo):.5f}")

            def on_slider(value, name=joint.name):
                if self._syncing:
                    return
                self._set_joint(name, float(value), source="slider")

            def on_text(value, name=joint.name):
                if self._syncing:
                    return
                try:
                    self._set_joint(name, float(value), source="text")
                except ValueError:
                    self.textboxes[name].set_val(f"{self.q.get(name, 0.0):.5f}")

            slider.on_changed(on_slider)
            textbox.on_submit(on_text)
            self.sliders[joint.name] = slider
            self.textboxes[joint.name] = textbox
            y -= row_h + gap

        closed_ax = self.fig.add_axes([0.18, 0.005, 0.1, 0.04])
        half_ax = self.fig.add_axes([0.30, 0.005, 0.1, 0.04])
        open_ax = self.fig.add_axes([0.42, 0.005, 0.1, 0.04])
        reset_ax = self.fig.add_axes([0.54, 0.005, 0.13, 0.04])
        for ax, label, callback in (
            (closed_ax, "Closed", lambda _event: self._set_fraction(0.0)),
            (half_ax, "Half", lambda _event: self._set_fraction(0.5)),
            (open_ax, "Open", lambda _event: self._set_fraction(1.0)),
            (reset_ax, "Reset View", lambda _event: self._reset_view()),
        ):
            button = self.Button(ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def _set_joint(self, name: str, value: float, source: str) -> None:
        slider = self.sliders[name]
        value = max(slider.valmin, min(slider.valmax, value))
        self.q[name] = value
        self._syncing = True
        try:
            if source != "slider":
                slider.set_val(value)
            if source != "text":
                self.textboxes[name].set_val(f"{value:.5f}")
        finally:
            self._syncing = False
        self._apply_pose(reset_limits=False)

    def _set_fraction(self, fraction: float) -> None:
        self._syncing = True
        try:
            for joint in self.joints:
                lo, hi = joint_limits(joint)
                value = lo + (hi - lo) * fraction
                self.q[joint.name] = value
                self.sliders[joint.name].set_val(value)
                self.textboxes[joint.name].set_val(f"{value:.5f}")
        finally:
            self._syncing = False
        self._apply_pose(reset_limits=False)

    def _reset_view(self) -> None:
        self.ax.view_init(elev=20, azim=-60)
        self._apply_pose(reset_limits=True)

    def _apply_pose(self, reset_limits: bool) -> None:
        transforms = uv.compute_link_transforms(self.robot, self.q)
        all_points = []
        for scatter, item in zip(self.scatters, self.items):
            points = transform_points(item["points"], transforms.get(item["link"], np.eye(4)))
            scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            all_points.append(points)

        for name, line in list(self.axis_lines.items()):
            line.remove()
            del self.axis_lines[name]
        self._draw_joint_axes(transforms)

        if reset_limits and all_points:
            equal_axes(self.ax, np.vstack(all_points))
        self.fig.canvas.draw_idle()

    def _draw_joint_axes(self, transforms: dict[str, np.ndarray]) -> None:
        for joint in self.joints:
            if joint.parent not in transforms:
                continue
            joint_world = transforms[joint.parent] @ joint.origin
            origin = joint_world[:3, 3]
            axis = joint_world[:3, :3] @ joint.axis
            norm = np.linalg.norm(axis)
            if norm <= 1e-12:
                continue
            axis = axis / norm
            length = 0.25
            pts = np.vstack([origin - axis * length, origin + axis * length])
            line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red", linewidth=2)
            self.axis_lines[joint.name] = line

    def show(self) -> None:
        self.plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Matplotlib URDF motion previewer.")
    parser.add_argument("urdf", nargs="?", default=str(uv._DEFAULT_URDF), help="URDF path")
    parser.add_argument("--open", type=float, default=0.0, help="initial open fraction for limited joints")
    parser.add_argument("--q", action="append", default=[], help="initial joint=value, can repeat or comma-separate")
    parser.add_argument("--max-points", type=int, default=9000, help="points sampled per visual mesh")
    return parser.parse_args()


def ensure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path.home() / ".config" / "matplotlib"
    if config_dir.exists() and os.access(config_dir, os.W_OK):
        return
    cache_dir = Path("/tmp") / f"urdf_motion_mpl_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def main() -> int:
    args = parse_args()
    ensure_matplotlib_cache()
    urdf_path = Path(os.path.expanduser(args.urdf)).resolve()
    if not urdf_path.exists():
        print(f"[urdf_motion_viewer_mpl] URDF not found: {urdf_path}", file=sys.stderr)
        return 2

    robot = uv.parse_urdf(urdf_path)
    q = uv.default_joint_values(robot, args.open, uv.parse_q(args.q))
    print(f"[urdf_motion_viewer_mpl] robot={robot.name} urdf={urdf_path}")
    for joint in movable_joints(robot):
        lo, hi = joint_limits(joint)
        print(f"[urdf_motion_viewer_mpl] joint {joint.name}: q={q.get(joint.name, 0.0):.5f} range=[{lo:.5f}, {hi:.5f}]")
    MatplotlibMotionViewer(robot, q, args.max_points).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
