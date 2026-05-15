#!/usr/bin/env python3
"""
Lightweight URDF previewer for reconstructed articulated objects.

The script intentionally implements only the URDF features needed for preview:
visual mesh geometry, visual origins, fixed/revolute/prismatic joints, joint
origins, axes, and limits. Mesh paths are resolved relative to the URDF file.

Examples:
    conda run -n owlsam python urdf_viewer.py
    conda run -n owlsam python urdf_viewer.py --open 1.0
    conda run -n owlsam python urdf_viewer.py --q cabinet_door_joint=0.4
    conda run -n owlsam python urdf_viewer.py --list --no-view
    conda run -n owlsam python urdf_viewer.py --export /tmp/cabinet_preview.glb --no-view

Interactive window:
    left drag   orbit camera
    right drag  pan
    wheel       zoom
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

_DEFAULT_URDF = Path.home() / (
    "projects/VLM_Planning_for_Articulated_Manipulation/"
    "recon_output/40147_1778791848/cabinet_drawer_rgbd.urdf"
)


@dataclass
class Visual:
    mesh_path: Path
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


@dataclass
class Link:
    name: str
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: list[Joint]
    urdf_path: Path


def xyz_attr(node: ET.Element | None, attr: str, default: tuple[float, float, float]) -> np.ndarray:
    if node is None or attr not in node.attrib:
        return np.array(default, dtype=float)
    values = [float(v) for v in node.attrib[attr].split()]
    if len(values) != 3:
        raise ValueError(f"{attr} must have 3 values, got {values}")
    return np.array(values, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_transform(node: ET.Element | None) -> np.ndarray:
    xyz = xyz_attr(node, "xyz", (0.0, 0.0, 0.0))
    rpy = xyz_attr(node, "rpy", (0.0, 0.0, 0.0))
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def axis_angle_transform(axis: np.ndarray, value: float) -> np.ndarray:
    import trimesh

    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(4)
    return trimesh.transformations.rotation_matrix(value, axis / norm)


def translation_transform(axis: np.ndarray, value: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    transform = np.eye(4)
    if norm > 1e-12:
        transform[:3, 3] = axis / norm * value
    return transform


def parse_urdf(urdf_path: Path) -> Robot:
    urdf_path = urdf_path.expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"not a URDF robot: root tag is {root.tag!r}")

    links: dict[str, Link] = {}
    for link_node in root.findall("link"):
        name = link_node.attrib["name"]
        link = Link(name=name)
        for visual_node in link_node.findall("visual"):
            mesh_node = visual_node.find("./geometry/mesh")
            if mesh_node is None or "filename" not in mesh_node.attrib:
                continue
            mesh_path = Path(mesh_node.attrib["filename"])
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            link.visuals.append(
                Visual(
                    mesh_path=mesh_path.resolve(),
                    transform=origin_transform(visual_node.find("origin")),
                    scale=xyz_attr(mesh_node, "scale", (1.0, 1.0, 1.0)),
                )
            )
        links[name] = link

    joints: list[Joint] = []
    for joint_node in root.findall("joint"):
        limit_node = joint_node.find("limit")
        lower = float(limit_node.attrib["lower"]) if limit_node is not None and "lower" in limit_node.attrib else None
        upper = float(limit_node.attrib["upper"]) if limit_node is not None and "upper" in limit_node.attrib else None
        joints.append(
            Joint(
                name=joint_node.attrib["name"],
                joint_type=joint_node.attrib.get("type", "fixed"),
                parent=joint_node.find("parent").attrib["link"],
                child=joint_node.find("child").attrib["link"],
                origin=origin_transform(joint_node.find("origin")),
                axis=xyz_attr(joint_node.find("axis"), "xyz", (1.0, 0.0, 0.0)),
                lower=lower,
                upper=upper,
            )
        )
    return Robot(name=root.attrib.get("name", urdf_path.stem), links=links, joints=joints, urdf_path=urdf_path)


def parse_q(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        for chunk in item.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"--q expects joint=value, got {chunk!r}")
            name, value = chunk.split("=", 1)
            result[name.strip()] = float(value)
    return result


def default_joint_values(robot: Robot, open_fraction: float, explicit_q: dict[str, float]) -> dict[str, float]:
    q: dict[str, float] = {}
    open_fraction = max(0.0, min(1.0, open_fraction))
    for joint in robot.joints:
        if joint.joint_type in ("fixed", "continuous"):
            q[joint.name] = 0.0
        elif joint.lower is not None and joint.upper is not None:
            q[joint.name] = joint.lower + (joint.upper - joint.lower) * open_fraction
        else:
            q[joint.name] = 0.0
    q.update(explicit_q)
    return q


def joint_motion(joint: Joint, value: float) -> np.ndarray:
    if joint.joint_type == "fixed":
        return np.eye(4)
    if joint.joint_type in ("revolute", "continuous"):
        return axis_angle_transform(joint.axis, value)
    if joint.joint_type == "prismatic":
        return translation_transform(joint.axis, value)
    print(f"[urdf_viewer] unsupported joint type {joint.joint_type!r}; treating as fixed", file=sys.stderr)
    return np.eye(4)


def compute_link_transforms(robot: Robot, q: dict[str, float]) -> dict[str, np.ndarray]:
    child_joints = {joint.child: joint for joint in robot.joints}
    roots = [name for name in robot.links if name not in child_joints]
    if not roots and robot.links:
        roots = [next(iter(robot.links))]

    children: dict[str, list[Joint]] = {}
    for joint in robot.joints:
        children.setdefault(joint.parent, []).append(joint)

    transforms: dict[str, np.ndarray] = {}

    def visit(link_name: str, parent_transform: np.ndarray) -> None:
        transforms[link_name] = parent_transform
        for joint in children.get(link_name, []):
            child_transform = parent_transform @ joint.origin @ joint_motion(joint, q.get(joint.name, 0.0))
            visit(joint.child, child_transform)

    for root in roots:
        visit(root, np.eye(4))
    return transforms


def load_visual_mesh(path: Path, scale: np.ndarray):
    import trimesh

    if not path.exists():
        raise FileNotFoundError(f"mesh not found: {path}")
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom.copy() for geom in loaded.geometry.values()]
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded.copy()
    if not np.allclose(scale, np.ones(3)):
        mesh.apply_scale(scale)
    return mesh


def color_for_link(index: int) -> np.ndarray:
    colors = np.array(
        [
            [176, 108, 82, 235],
            [86, 143, 196, 235],
            [105, 166, 117, 235],
            [196, 156, 72, 235],
            [151, 116, 181, 235],
        ],
        dtype=np.uint8,
    )
    return colors[index % len(colors)]


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, color: tuple[int, int, int, int]):
    import trimesh

    vector = end - start
    height = float(np.linalg.norm(vector))
    if height <= 1e-12:
        return None
    transform = trimesh.geometry.align_vectors([0, 0, 1], vector / height)
    transform[:3, 3] = (start + end) * 0.5
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=24, transform=transform)
    cyl.visual.face_colors = np.array(color, dtype=np.uint8)
    return cyl


def add_joint_markers(scene, robot: Robot, link_transforms: dict[str, np.ndarray], q: dict[str, float]) -> None:
    import trimesh

    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all():
        length = 0.5
    else:
        length = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.08, 0.08)
    radius = length * 0.035
    for joint in robot.joints:
        if joint.joint_type == "fixed" or joint.parent not in link_transforms:
            continue
        joint_world = link_transforms[joint.parent] @ joint.origin
        origin = joint_world[:3, 3]
        axis = joint_world[:3, :3] @ joint.axis
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            continue
        axis = axis / norm
        marker = cylinder_between(origin - axis * length, origin + axis * length, radius, (230, 50, 45, 255))
        if marker is not None:
            scene.add_geometry(marker, node_name=f"{joint.name}_axis")
        sphere = trimesh.creation.uv_sphere(radius=radius * 2.4)
        sphere.visual.face_colors = np.array([255, 230, 60, 255], dtype=np.uint8)
        scene.add_geometry(sphere, node_name=f"{joint.name}_origin", transform=joint_world)
        print(
            f"[urdf_viewer] joint {joint.name}: type={joint.joint_type} "
            f"q={q.get(joint.name, 0.0):.4f} origin={origin.round(4)} axis={axis.round(4)}"
        )


def build_scene(robot: Robot, q: dict[str, float], show_axes: bool = True):
    import trimesh

    scene = trimesh.Scene()
    link_transforms = compute_link_transforms(robot, q)
    for link_index, (link_name, link) in enumerate(robot.links.items()):
        world = link_transforms.get(link_name, np.eye(4))
        for visual_index, visual in enumerate(link.visuals):
            mesh = load_visual_mesh(visual.mesh_path, visual.scale)
            if not hasattr(mesh.visual, "kind") or mesh.visual.kind in (None, "none"):
                mesh.visual.face_colors = color_for_link(link_index)
            scene.add_geometry(
                mesh,
                node_name=f"{link_name}_{visual_index}",
                geom_name=f"{link_name}_{visual.mesh_path.name}",
                transform=world @ visual.transform,
            )
    if show_axes:
        add_joint_markers(scene, robot, link_transforms, q)
    return scene


def print_summary(robot: Robot, q: dict[str, float]) -> None:
    print(f"[urdf_viewer] robot: {robot.name}")
    print(f"[urdf_viewer] urdf:  {robot.urdf_path}")
    print("[urdf_viewer] links:")
    for link in robot.links.values():
        visuals = ", ".join(str(v.mesh_path.relative_to(robot.urdf_path.parent)) for v in link.visuals) or "(no visual)"
        print(f"  - {link.name}: {visuals}")
    print("[urdf_viewer] joints:")
    for joint in robot.joints:
        lim = ""
        if joint.lower is not None or joint.upper is not None:
            lim = f" limits=[{joint.lower}, {joint.upper}]"
        print(
            f"  - {joint.name}: {joint.joint_type} {joint.parent} -> {joint.child} "
            f"axis={joint.axis.tolist()} q={q.get(joint.name, 0.0):.4f}{lim}"
        )


def export_scene(scene, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("glb", "gltf", "obj", "stl", "ply"):
        scene.export(path)
    else:
        raise ValueError(f"unsupported export suffix {path.suffix!r}; use .glb, .gltf, .obj, .stl, or .ply")
    print(f"[urdf_viewer] exported {path}")


def trimesh_to_open3d(mesh):
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(vertices):
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
        out.vertex_colors = o3d.utility.Vector3dVector(colors)
    elif hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors):
        color = np.asarray(mesh.visual.face_colors[0, :3], dtype=np.float64) / 255.0
        out.paint_uniform_color(color)
    else:
        out.paint_uniform_color([0.72, 0.72, 0.72])
    out.compute_vertex_normals()
    return out


def show_open3d(scene) -> None:
    import open3d as o3d

    geometries = []
    for geom in scene.dump(concatenate=False):
        if hasattr(geom, "vertices") and hasattr(geom, "faces") and len(geom.faces):
            geometries.append(trimesh_to_open3d(geom))
    if not geometries:
        raise RuntimeError("scene has no triangle meshes to display")

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    geometries.append(coord)
    o3d.visualization.draw_geometries(geometries, window_name="URDF preview")


def show_scene(scene, backend: str) -> None:
    if backend in ("auto", "open3d"):
        try:
            show_open3d(scene)
            return
        except Exception as exc:
            if backend == "open3d":
                raise
            print(f"[urdf_viewer] open3d viewer failed, trying trimesh: {exc}", file=sys.stderr)

    try:
        scene.show()
    except ImportError as exc:
        if "pyglet<2" in str(exc):
            raise ImportError(
                "The trimesh windowed viewer needs pyglet<2 in this environment. "
                "Use the default Open3D backend, or run with --export /tmp/preview.glb --no-view."
            ) from exc
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a reconstructed URDF with trimesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("urdf", nargs="?", default=str(_DEFAULT_URDF), help="URDF path")
    parser.add_argument("--open", type=float, default=0.0, help="open fraction for limited joints, 0=lower, 1=upper")
    parser.add_argument("--q", action="append", default=[], help="joint=value, can repeat or comma-separate")
    parser.add_argument("--no-axes", action="store_true", help="hide joint axis/origin markers")
    parser.add_argument("--list", action="store_true", help="print links, joints, limits, and selected q")
    parser.add_argument("--export", type=Path, help="export posed preview scene (.glb/.gltf/.obj/.stl/.ply)")
    parser.add_argument("--no-view", action="store_true", help="do not open the interactive viewer")
    parser.add_argument("--backend", choices=("auto", "open3d", "trimesh"), default="auto", help="interactive viewer backend")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = Path(os.path.expanduser(args.urdf)).resolve()
    if not urdf_path.exists():
        print(f"[urdf_viewer] URDF not found: {urdf_path}", file=sys.stderr)
        return 2

    robot = parse_urdf(urdf_path)
    q = default_joint_values(robot, args.open, parse_q(args.q))
    if args.list:
        print_summary(robot, q)

    scene = build_scene(robot, q, show_axes=not args.no_axes)
    print(f"[urdf_viewer] loaded {len(robot.links)} link(s), {len(robot.joints)} joint(s), {len(scene.geometry)} scene geometries")

    if args.export:
        export_scene(scene, args.export)
    if not args.no_view:
        show_scene(scene, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
