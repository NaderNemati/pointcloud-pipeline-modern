#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-quality offscreen renderer for LiDAR frames (Open3D).

Fixes: too-far camera, tiny points, dark colors.
- Robust global camera (quantile-clipped), or focus from OBBs.
- Stable camera + optional Z-azimuth orbit.
- MSAA/TAA, ACES tone-mapping, optional AO.
- Color gain/gamma to brighten sparsely colored points.
- Optional multi-frame accumulation (blends K previous frames).

Docs (APIs): OffscreenRenderer.setup_camera (look-at overload),
View post-processing/AA/AO/ColorGrading, Open3DScene set_background/set_lighting.
"""

import os, glob, math, argparse, re
import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw

# -------------------------- utils --------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(s))]

def list_frames(seq_dir: str):
    frames = sorted(
        [p for p in glob.glob(os.path.join(seq_dir, "*.ply"))
         if "_box_" not in os.path.basename(p)],
        key=natural_key
    )
    return frames

def per_frame_boxes(seq_dir: str, stem: str):
    return sorted(glob.glob(os.path.join(seq_dir, f"{stem}_box_*.ply")), key=natural_key)

def has_colors(pcd: o3d.geometry.PointCloud) -> bool:
    return len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0

def apply_color_gain_gamma(pcd, gain=1.0, gamma=1.0):
    """c' = clamp((c*gain) ** gamma) for each channel."""
    if not has_colors(pcd):
        return
    c = np.asarray(pcd.colors, dtype=np.float32)
    c = np.clip((c * float(gain)) ** float(gamma), 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(c)

def as_material_point(size=2.0):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"    # preserve vertex colors; lighting not needed for points
    mat.point_size = float(size)   # pixel size on screen
    return mat

def as_material_line(width=2.0, color=(1.0, 0.85, 0.2, 1.0)):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.line_width = float(width)
    mat.base_color = color
    return mat

# -------------------------- camera helpers --------------------------

def robust_xyz_quantiles(pcds, q_lo=0.01, q_hi=0.99):
    """Compute robust min/max per-axis from a set of point clouds (ignore outliers)."""
    pts = []
    for pc in pcds:
        if len(pc.points) == 0: 
            continue
        pts.append(np.asarray(pc.points))
    if not pts:
        raise RuntimeError("No points to compute quantiles.")
    P = np.concatenate(pts, axis=0)
    lo = np.quantile(P, q_lo, axis=0)
    hi = np.quantile(P, q_hi, axis=0)
    return lo, hi

def aabb_from_lo_hi(lo, hi):
    aabb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    return aabb

def union_aabb(a, b):
    return a + b if a is not None else b

def orbit_eye_around_Z(center, eye, orbit_deg):
    t = math.radians(orbit_deg)
    R = np.array([[ math.cos(t), -math.sin(t), 0.0],
                  [ math.sin(t),  math.cos(t), 0.0],
                  [ 0.0,          0.0,         1.0 ]], dtype=np.float32)
    return (R @ (eye - center)) + center

def setup_lookat(renderer, fov_deg, center, eye, up):
    # OffscreenRenderer.setup_camera supports look-at overload. :contentReference[oaicite:1]{index=1}
    renderer.setup_camera(float(fov_deg),
                          np.asarray(center, dtype=np.float32),
                          np.asarray(eye, dtype=np.float32),
                          np.asarray(up, dtype=np.float32))

def compute_focus_aabb_from_boxes(seq_dir, sample_frames):
    """If OBB LineSets exist, use their union AABB as focus (tighter than raw points)."""
    focus = None
    for fp in sample_frames:
        stem = os.path.splitext(os.path.basename(fp))[0]
        for b in per_frame_boxes(seq_dir, stem):
            try:
                ls = o3d.io.read_line_set(b)
                focus = union_aabb(focus, ls.get_axis_aligned_bounding_box())
            except Exception:
                pass
    return focus

def compute_global_camera(seq_dir, frames, fov_deg=50.0, method="robust", qlo=0.01, qhi=0.99,
                          z_clip=None, use_boxes=False):
    """
    Build ONE look-at camera:
    - method 'robust': quantile-clipped AABB over 3 sampled frames (0, mid, last)
    - use_boxes: if True and *_box_*.ply exist, focus on their union instead (tighter)
    - z_clip=(zmin,zmax): optional vertical clamp before AABB (helps parking lots)
    """
    if not frames:
        raise RuntimeError("No frames to compute camera.")
    picks = [frames[0]]
    if len(frames) >= 3:
        picks = [frames[0], frames[len(frames)//2], frames[-1]]

    if use_boxes:
        aabb = compute_focus_aabb_from_boxes(seq_dir, picks)
        if aabb is None:
            # fall back to robust if no boxes found
            use_boxes = False

    if not use_boxes:
        pcs = []
        for p in picks:
            pc = o3d.io.read_point_cloud(p)
            if z_clip is not None and len(pc.points):
                pts = np.asarray(pc.points)
                zmin, zmax = z_clip
                mask = (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
                pc = pc.select_by_index(np.nonzero(mask)[0])
            pcs.append(pc)
        lo, hi = robust_xyz_quantiles(pcs, q_lo=qlo, q_hi=qhi)
        aabb = aabb_from_lo_hi(lo, hi)

    center = aabb.get_center()
    extent = float(np.linalg.norm(aabb.get_extent()))
    dist = 1.4 * extent if extent > 0 else 30.0

    up  = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eye = center + np.array([0.0, -dist, 0.55 * dist], dtype=np.float32)
    return float(fov_deg), center, eye, up

# -------------------------- renderer config --------------------------

def configure_view_quality(scene: rendering.Open3DScene,
                           msaa=8, taa=True, ao=False, bg=(0,0,0,1)):
    # Scene background & simple lighting profile. :contentReference[oaicite:2]{index=2}
    try:
        scene.set_background(np.array(bg, dtype=np.float32))
    except Exception:
        pass
    try:
        scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS,
                           np.array([0.577, -0.577, -0.577], dtype=np.float32))
    except Exception:
        pass

    # View post-processing and AA/AO. :contentReference[oaicite:3]{index=3}
    view = scene.view
    for fn, args in [
        ("set_post_processing", (True,)),
        ("set_sample_count", (int(msaa),)),
    ]:
        try:
            getattr(view, fn)(*args)
        except Exception:
            pass
    try:
        view.set_antialiasing(True, temporal=bool(taa))
    except Exception:
        try:
            view.set_antialiasing(True)
        except Exception:
            pass
    try:
        view.set_ambient_occlusion(bool(ao))
    except Exception:
        pass

    # ACES tone mapping. :contentReference[oaicite:4]{index=4}
    try:
        from open3d.visualization.rendering import ColorGrading
        cg = ColorGrading(ColorGrading.Quality.HIGH,
                          ColorGrading.ToneMapping.ACES)
        view.set_color_grading(cg)
    except Exception:
        pass

# -------------------------- main render --------------------------

def render_sequence(seq_dir,
                    out_dir,
                    width=1920, height=1080, fps=15,
                    orbit_deg_per_frame=0.0,
                    point_size=3.0, line_width=2.0,
                    bg=(0,0,0,1),
                    stable_camera=True,
                    robust=True, qlo=0.01, qhi=0.99,
                    z_clip=None, focus_boxes=True,
                    msaa=8, taa=True, ao=False,
                    color_gain=1.3, color_gamma=0.9,
                    accum_frames=0,            # 0 = off; try 3–5 for density
                    max_frames=None, hud=True):
    os.makedirs(out_dir, exist_ok=True)
    frames = list_frames(seq_dir)
    if not frames:
        raise SystemExit(f"No frame PLYs found in {seq_dir}")

    if max_frames is not None:
        frames = frames[:max_frames]

    renderer = rendering.OffscreenRenderer(int(width), int(height))
    configure_view_quality(renderer.scene, msaa=msaa, taa=taa, ao=ao, bg=bg)

    mat_p = as_material_point(point_size)
    mat_l = as_material_line(line_width)

    # Camera once (stable) using robust quantiles or boxes
    if stable_camera:
        fov, c0, e0, u0 = compute_global_camera(
            seq_dir, frames,
            fov_deg=50.0,
            method="robust" if robust else "bbox",
            qlo=qlo, qhi=qhi,
            z_clip=z_clip,
            use_boxes=focus_boxes
        )
    else:
        fov, c0, e0, u0 = 55.0, None, None, None  # computed per-frame

    # For accumulation
    history = []

    for idx, fpath in enumerate(frames):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        renderer.scene.clear_geometry()

        # base frame
        pcd = o3d.io.read_point_cloud(fpath)
        if color_gain != 1.0 or color_gamma != 1.0:
            apply_color_gain_gamma(pcd, gain=color_gain, gamma=color_gamma)
        renderer.scene.add_geometry(f"pc_{stem}", pcd, mat_p)

        # multi-frame accumulation (add a few previous clouds, lighter)
        if accum_frames > 0:
            history.append(pcd)
            if len(history) > accum_frames:
                history.pop(0)
            for j, prev in enumerate(history[:-1]):
                if has_colors(prev):
                    # fade older frames
                    c = np.asarray(prev.colors, dtype=np.float32)
                    fade = 0.6 * ((j + 1) / (accum_frames))
                    prev.colors = o3d.utility.Vector3dVector(np.clip(c * (1.0 - fade), 0, 1))
                renderer.scene.add_geometry(f"pc_prev_{idx}_{j}", prev, mat_p)

        # OBBs for this frame (draw after so they appear on top)
        for bpath in per_frame_boxes(seq_dir, stem):
            try:
                ls = o3d.io.read_line_set(bpath)
                renderer.scene.add_geometry(f"box_{os.path.basename(bpath)}", ls, mat_l)
            except Exception:
                pass

        # Camera
        if stable_camera:
            eye_eff = e0
            if orbit_deg_per_frame != 0.0:
                eye_eff = orbit_eye_around_Z(c0, e0, idx * orbit_deg_per_frame)
            setup_lookat(renderer, fov, c0, eye_eff, u0)
        else:
            # per-frame robust fit (slower)
            bb = pcd.get_axis_aligned_bounding_box()
            center = bb.get_center()
            extent = float(np.linalg.norm(bb.get_extent()))
            dist = 1.5 * extent if extent > 0 else 30.0
            eye = center + np.array([0.0, -dist, 0.55*dist], dtype=np.float32)
            if orbit_deg_per_frame != 0.0:
                eye = orbit_eye_around_Z(center, eye, idx * orbit_deg_per_frame)
            setup_lookat(renderer, 55.0, center, eye, np.array([0,0,1.0], dtype=np.float32))

        # Render → image
        img = renderer.render_to_image()    # Open3D Image (RGBA). :contentReference[oaicite:5]{index=5}
        pil = Image.fromarray(np.asarray(img))
        if hud:
            draw = ImageDraw.Draw(pil)
            draw.text((16, 16), f"{stem}", fill=(255,255,255,220))
        pil.save(os.path.join(out_dir, f"{idx:06d}.png"))

    print(f"Saved {len(frames)} PNGs to {out_dir}")
    print("Tip: downscale with Lanczos when encoding for extra sharpness, e.g.:")
    print(f"  ffmpeg -y -framerate {fps} -i {out_dir}/%06d.png "
          f"-vf \"scale=1920:-2:flags=lanczos\" "
          f"-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p {os.path.dirname(out_dir)}/video.mp4")

# -------------------------- CLI --------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="seq_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=0.0, help="deg per frame about Z")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--line-width", type=float, default=2.0)
    ap.add_argument("--bg", type=float, nargs=4, default=(0,0,0,1), help="RGBA bg")
    ap.add_argument("--stable-camera", action="store_true", help="Use one camera for all frames")
    ap.add_argument("--no-robust", action="store_true", help="Disable quantile clipping")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.01, 0.99), help="q_lo q_hi for robust fit")
    ap.add_argument("--z", type=float, nargs=2, default=None, help="zmin zmax (vertical clamp)")
    ap.add_argument("--no-focus-boxes", action="store_true", help="Ignore *_box_*.ply for focus")
    ap.add_argument("--msaa", type=int, default=8)
    ap.add_argument("--no-taa", action="store_true")
    ap.add_argument("--ao", action="store_true")
    ap.add_argument("--gain", type=float, default=1.3, help="color gain")
    ap.add_argument("--gamma", type=float, default=0.9, help="color gamma (<1 brightens mids)")
    ap.add_argument("--accum", type=int, default=0, help="# of previous frames to overlay")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-hud", action="store_true")
    args = ap.parse_args()

    render_sequence(
        seq_dir=args.seq_dir,
        out_dir=args.out_dir,
        width=args.w, height=args.h, fps=args.fps,
        orbit_deg_per_frame=args.orbit,
        point_size=args.point_size, line_width=args.line_width,
        bg=tuple(args.bg),
        stable_camera=bool(args.stable_camera),
        robust=(not args.no_robust), qlo=args.clip[0], qhi=args.clip[1],
        z_clip=tuple(args.z) if args.z else None,
        focus_boxes=(not args.no_focus_boxes),
        msaa=args.msaa, taa=(not args.no_taa), ao=args.ao,
        color_gain=args.gain, color_gamma=args.gamma,
        accum_frames=max(0, args.accum),
        max_frames=args.max_frames, hud=(not args.no_hud)
    )

