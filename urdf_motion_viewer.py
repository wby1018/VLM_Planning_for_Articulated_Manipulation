#!/usr/bin/env python3
"""
Interactive URDF motion preview window.

The left side is an Open3D 3D view. Drag in the viewport to orbit, right-drag
to pan, and use the mouse wheel to zoom. The right panel exposes every movable
URDF joint as both a slider and a numeric input; changing either immediately
updates the posed model.

Examples:
    conda run -n owlsam python urdf_motion_viewer.py
    conda run -n owlsam python urdf_motion_viewer.py --open 1.0
    conda run -n owlsam python urdf_motion_viewer.py --q cabinet_door_joint=0.4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import urdf_viewer as uv

_WINDOW_REF = None


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


class MotionWindow:
    def __init__(self, robot: uv.Robot, q: dict[str, float], width: int, height: int):
        self.robot = robot
        self.q = dict(q)
        self.joints = movable_joints(robot)
        self.visual_nodes: list[tuple[str, str, np.ndarray]] = []
        self.controls: dict[str, tuple[gui.Slider, gui.NumberEdit, gui.Label]] = {}
        self._syncing_controls = False

        self.window = gui.Application.instance.create_window("URDF Motion Viewer", width, height)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.show_axes(True)
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            [0.45, -0.35, -0.82],
        )

        self.panel = self._build_panel()
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        self._load_visuals()
        self._apply_pose(reset_camera=True)

    def _guard(self, func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                print(f"[urdf_motion_viewer] callback failed: {exc}", file=sys.stderr)
                return None

        return wrapped

    def _build_panel(self):
        em = self.window.theme.font_size
        panel = gui.ScrollableVert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))

        title = gui.Label("URDF Motion")
        panel.add_child(title)
        panel.add_child(gui.Label(self.robot.urdf_path.name))
        panel.add_fixed(0.4 * em)

        buttons = gui.Horiz(0.35 * em)
        closed_btn = gui.Button("Closed")
        closed_btn.set_on_clicked(self._guard(lambda: self._set_fraction(0.0)))
        mid_btn = gui.Button("Half")
        mid_btn.set_on_clicked(self._guard(lambda: self._set_fraction(0.5)))
        open_btn = gui.Button("Open")
        open_btn.set_on_clicked(self._guard(lambda: self._set_fraction(1.0)))
        buttons.add_child(closed_btn)
        buttons.add_child(mid_btn)
        buttons.add_child(open_btn)
        panel.add_child(buttons)

        reset_camera_btn = gui.Button("Reset Camera")
        reset_camera_btn.set_on_clicked(self._guard(lambda: self._reset_camera()))
        panel.add_child(reset_camera_btn)
        panel.add_fixed(0.7 * em)

        if not self.joints:
            panel.add_child(gui.Label("No movable joints"))
            return panel

        for joint in self.joints:
            lo, hi = joint_limits(joint)
            panel.add_child(gui.Label(f"{joint.name}  ({joint.joint_type})"))

            value_label = gui.Label("")
            panel.add_child(value_label)

            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(lo, hi)
            slider.double_value = float(self.q.get(joint.name, lo))

            number = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            number.set_limits(lo, hi)
            number.decimal_precision = 5
            number.set_preferred_width(8 * em)
            number.set_value(float(self.q.get(joint.name, lo)))

            def on_slider(value, name=joint.name):
                if self._syncing_controls:
                    return
                self._set_joint_value(name, float(value), source="slider")

            def on_number(value, name=joint.name):
                if self._syncing_controls:
                    return
                self._set_joint_value(name, float(value), source="number")

            slider.set_on_value_changed(self._guard(on_slider))
            number.set_on_value_changed(self._guard(on_number))

            row = gui.Horiz(0.35 * em)
            row.add_child(slider)
            row.add_child(number)
            panel.add_child(row)

            axis = np.asarray(joint.axis, dtype=float)
            panel.add_child(gui.Label(f"axis [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]"))
            panel.add_child(gui.Label(f"range [{lo:.5f}, {hi:.5f}]"))
            panel.add_fixed(0.6 * em)
            self.controls[joint.name] = (slider, number, value_label)

        self._refresh_value_labels()
        return panel

    def _on_layout(self, layout_context):
        rect = self.window.content_rect
        panel_width = int(23 * layout_context.theme.font_size)
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

    def _load_visuals(self) -> None:
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.82, 0.82, 0.82, 1.0]

        for link_name, link in self.robot.links.items():
            for visual_index, visual in enumerate(link.visuals):
                mesh = uv.load_visual_mesh(visual.mesh_path, visual.scale)
                geom = uv.trimesh_to_open3d(mesh)
                node_name = f"{link_name}_{visual_index}"
                self.scene_widget.scene.add_geometry(node_name, geom, material)
                self.visual_nodes.append((link_name, node_name, visual.transform))

    def _apply_pose(self, reset_camera: bool = False) -> None:
        transforms = uv.compute_link_transforms(self.robot, self.q)
        for link_name, node_name, visual_transform in self.visual_nodes:
            world = transforms.get(link_name, np.eye(4)) @ visual_transform
            self.scene_widget.scene.set_geometry_transform(node_name, world)
        self._refresh_value_labels()
        self.scene_widget.force_redraw()
        if reset_camera:
            self._reset_camera()

    def _reset_camera(self) -> None:
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        self.scene_widget.look_at(bounds.get_center(), bounds.get_center() + [0, -3, 1.2], [0, 0, 1])

    def _set_joint_value(self, name: str, value: float, source: str) -> None:
        slider, number, _label = self.controls[name]
        lo, hi = slider.get_minimum_value(), slider.get_maximum_value()
        value = max(lo, min(hi, value))
        self.q[name] = value

        self._syncing_controls = True
        try:
            if source != "slider":
                slider.double_value = value
            if source != "number":
                number.set_value(value)
        finally:
            self._syncing_controls = False
        self._apply_pose()

    def _set_fraction(self, fraction: float) -> None:
        self._syncing_controls = True
        try:
            for joint in self.joints:
                lo, hi = joint_limits(joint)
                value = lo + (hi - lo) * fraction
                self.q[joint.name] = value
                slider, number, _label = self.controls[joint.name]
                slider.double_value = value
                number.set_value(value)
        finally:
            self._syncing_controls = False
        self._apply_pose()

    def _refresh_value_labels(self) -> None:
        for joint in self.joints:
            if joint.name not in self.controls:
                continue
            _slider, _number, label = self.controls[joint.name]
            value = self.q.get(joint.name, 0.0)
            if joint.joint_type in ("revolute", "continuous"):
                label.text = f"q = {value:.5f} rad  ({np.degrees(value):.2f} deg)"
            else:
                label.text = f"q = {value:.5f} m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a draggable URDF motion preview window with joint sliders and numeric inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("urdf", nargs="?", default=str(uv._DEFAULT_URDF), help="URDF path")
    parser.add_argument("--open", type=float, default=0.0, help="initial open fraction for limited joints")
    parser.add_argument("--q", action="append", default=[], help="initial joint=value, can repeat or comma-separate")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    return parser.parse_args()


def main() -> int:
    global _WINDOW_REF
    args = parse_args()
    urdf_path = Path(os.path.expanduser(args.urdf)).resolve()
    if not urdf_path.exists():
        print(f"[urdf_motion_viewer] URDF not found: {urdf_path}", file=sys.stderr)
        return 2

    robot = uv.parse_urdf(urdf_path)
    q = uv.default_joint_values(robot, args.open, uv.parse_q(args.q))
    print(f"[urdf_motion_viewer] robot={robot.name} urdf={urdf_path}")
    for joint in movable_joints(robot):
        lo, hi = joint_limits(joint)
        print(f"[urdf_motion_viewer] joint {joint.name}: type={joint.joint_type} q={q.get(joint.name, 0.0):.5f} range=[{lo:.5f}, {hi:.5f}]")

    gui.Application.instance.initialize()
    _WINDOW_REF = MotionWindow(robot, q, args.width, args.height)
    gui.Application.instance.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
