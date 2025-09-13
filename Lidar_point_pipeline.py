#!/usr/bin/env python3
"""
LiDAR Million-Point Pipeline (modern variant)
===========================================

This script implements an end-to-end LiDAR point cloud pipeline inspired by the
post you shared — but with some *newer* twists:

• Range‑adaptive voxel downsampling (preserves near-field detail, thins far‑field)
• Robust normal estimation (KD‑Tree) to aid ground separation
• Ground removal via (a) SemanticKITTI labels if available, else (b) CSF Cloth Simulation
  filter if installed (pip install cloth-simulation-filter), else (c) RANSAC plane(s)
• Instance discovery with HDBSCAN (soft clustering w/ probabilities)
• PCA‑based Oriented Bounding Boxes (OBB) per cluster
• Optional Open3D visualization and JSON export per frame

The pipeline works on KITTI/ SemanticKITTI layout and on bare .bin files.

Usage (examples)
----------------
1) Process SemanticKITTI sequence 08 (first 100 frames), using labels if present:
   python lidar_pipeline.py \
       --dataset_root /path/to/kitti/dataset \
       --sequence 08 \
       --start 0 --end 100 \
       --use-labels \
       --out ./outputs/seq08

2) Same, but force classical ground removal via CSF (cloth simulation) if installed:
   python lidar_pipeline.py --dataset_root /path/to/kitti/dataset --sequence 08 \
       --start 0 --end 50 --no-use-labels --use-csf

3) Single .bin file (KITTI velodyne scan):
   python lidar_pipeline.py --bin ./000123.bin --out ./outputs/single

Install deps
------------
python -m pip install --upgrade numpy open3d hdbscan tqdm pyyaml
# Optional for CSF ground filter
python -m pip install cloth-simulation-filter

Folder assumptions (SemanticKITTI)
----------------------------------
<dataset_root>/sequences/<seq>/velodyne/000000.bin
<dataset_root>/sequences/<seq>/labels/  000000.label  (optional)

Notes
-----
• SemanticKITTI .label packs instance/semantic as 32‑bit uint: low 16 bits = semantic, high 16 bits = instance.
• Ground semantic ids (raw) commonly used: {40: road, 44: parking, 48: sidewalk, 49: other-ground, 72: terrain, 60: lane-marking}.
  You can override via --ground-ids.
• HDBSCAN is density‑based and returns -1 for noise and per-point membership probabilities.
• OBBs are PCA-based boxes from Open3D; fields: center (x,y,z), extent (dx,dy,dz), yaw (around +Z).

Outputs
-------
• For each processed frame: JSON with clusters + boxes under <out>/sequenceXX/000000.json
• Optionally PLYs for colored point clouds and LineSets for OBBs (if --save-vis)

"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# --- Optional heavy deps ---
try:
    import open3d as o3d  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Open3D is required: pip install open3d")

# HDBSCAN can be sklearn's experimental or the community package; prefer community package
try:
    import hdbscan  # type: ignore
    HDBSCAN = hdbscan.HDBSCAN
except Exception:
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore
    except Exception:
        raise SystemExit("HDBSCAN is required: pip install hdbscan or sklearn>=1.4")

# CSF cloth simulation filter: optional
try:
    import CSF  # from cloth-simulation-filter
    HAS_CSF = True
except Exception:
    HAS_CSF = False

# ------------------------------
# Data structures
# ------------------------------
@dataclass
class ClusterBox:
    cluster_id: int
    n_points: int
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    yaw: float  # radians, rotation around Z
    prob_mean: float  # mean HDBSCAN soft-membership

@dataclass
class FrameResult:
    sequence: str
    frame_idx: int
    n_points_in: int
    n_points_proc: int
    n_points_nonground: int
    n_clusters: int
    clusters: List[ClusterBox]

# ------------------------------
# IO helpers (KITTI / SemanticKITTI)
# ------------------------------

def read_kitti_bin(bin_path: str | Path) -> np.ndarray:
    """Read KITTI/Velodyne .bin -> (N,4) float32 [x,y,z,intensity]."""
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Unexpected .bin size: {bin_path}")
    return arr.reshape(-1, 4)


def read_semantic_label(label_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read SemanticKITTI .label -> (semantic_id uint16, instance_id uint16)."""
    raw = np.fromfile(str(label_path), dtype=np.uint32)
    sem = raw & 0xFFFF
    inst = raw >> 16
    return sem.astype(np.uint16), inst.astype(np.uint16)


# ------------------------------
# Geometry helpers
# ------------------------------

def to_o3d(points_xyz: np.ndarray, intens: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64, copy=False))
    if intens is not None:
        # grayscale by intensity
        s = intens.astype(np.float64)
        s = (s - s.min()) / (s.ptp() + 1e-9)
        colors = np.stack([s, s, s], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def estimate_normals_inplace(pcd: o3d.geometry.PointCloud, radius: float = 0.7, max_nn: int = 30) -> None:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))


def adaptive_range_voxel_downsample(points: np.ndarray,
                                    bins: Iterable[float] = (15.0, 30.0, 60.0, 120.0),
                                    voxel_sizes: Iterable[float] = (0.05, 0.10, 0.20, 0.40)) -> np.ndarray:
    """Range-adaptive voxel downsample: finer near, coarser far.
    Returns (M,3) xyz only (attributes discarded).
    """
    xyz = points[:, :3]
    d_xy = np.linalg.norm(xyz[:, :2], axis=1)
    bins = list(bins)
    voxel_sizes = list(voxel_sizes)
    keep_xyz = []
    for i, rmax in enumerate(bins):
        rmin = 0.0 if i == 0 else bins[i - 1]
        mask = (d_xy >= rmin) & (d_xy < rmax)
        if not np.any(mask):
            continue
        pcd = to_o3d(xyz[mask])
        ds = pcd.voxel_down_sample(voxel_sizes[i])
        if len(ds.points) > 0:
            keep_xyz.append(np.asarray(ds.points))
    # far tail beyond last bin
    mask_tail = d_xy >= bins[-1]
    if np.any(mask_tail):
        pcd = to_o3d(xyz[mask_tail])
        ds = pcd.voxel_down_sample(voxel_sizes[-1] * 1.2)
        if len(ds.points) > 0:
            keep_xyz.append(np.asarray(ds.points))
    if len(keep_xyz) == 0:
        return xyz
    return np.concatenate(keep_xyz, axis=0)


def ransac_ground_mask(pcd: o3d.geometry.PointCloud,
                       distance_thresh: float = 0.2,
                       max_planes: int = 2,
                       min_inliers: int = 20000) -> np.ndarray:
    """Segment one or more dominant horizontal planes as ground; returns boolean mask for ground points.
    Heuristic: accept planes with normal close to +Z and many inliers.
    """
    all_idx = np.arange(len(pcd.points))
    remaining = pcd
    ground_idx = []
    for _ in range(max_planes):
        if len(remaining.points) < min_inliers:
            break
        plane_model, inliers = remaining.segment_plane(distance_threshold=distance_thresh,
                                                      ransac_n=3,
                                                      num_iterations=200)
        # plane normal
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-9)
        # keep only near-horizontal
        if abs(normal[2]) < 0.85:  # not horizontal enough
            break
        if len(inliers) < min_inliers:
            break
        # map inliers back to original indices
        inliers = np.array(inliers, dtype=int)
        remaining_idx = np.array(remaining.indices) if hasattr(remaining, 'indices') else all_idx
        ground_idx.append(remaining_idx[inliers])
        # carve them out
        not_in = np.ones(len(remaining.points), dtype=bool)
        not_in[inliers] = False
        tmp = remaining.select_by_index(inliers, invert=True)
        # carry indices
        tmp.indices = remaining_idx[not_in]
        remaining = tmp
    if ground_idx:
        ground_idx = np.concatenate(ground_idx, axis=0)
        mask = np.zeros(len(pcd.points), dtype=bool)
        mask[ground_idx] = True
        return mask
    return np.zeros(len(pcd.points), dtype=bool)


def csf_ground_mask(xyz: np.ndarray,
                    cloth_res: float = 0.5,
                    rigidness: int = 3,
                    slope_smooth: bool = False) -> np.ndarray:
    """Cloth Simulation Filter ground mask; requires cloth-simulation-filter. Returns boolean mask for ground points."""
    if not HAS_CSF:
        raise RuntimeError("CSF not available; pip install cloth-simulation-filter")
    csf = CSF.CSF()
    # parameters (see PDAL/CSF docs). Keep it simple; tweak as needed.
    csf.params.bSloopSmooth = bool(slope_smooth)
    csf.params.cloth_resolution = float(cloth_res)
    csf.params.rigidness = int(rigidness)
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.5
    # set cloud
    csf.setPointCloud(xyz.astype(np.float64).tolist())
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    if len(ground_idx) > 0:
        mask[np.array(list(ground_idx), dtype=int)] = True
    return mask


def hdbscan_cluster(xyz: np.ndarray,
                    min_cluster_size: int = 40,
                    min_samples: Optional[int] = None,
                    leaf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run HDBSCAN on XY (optionally XYZ). Returns labels and soft probabilities."""
    feats = xyz[:, :2]  # BEV clustering is often more stable
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples if min_samples is not None else min_cluster_size,
                        cluster_selection_method=('leaf' if leaf else 'eom'))
    labels = clusterer.fit_predict(feats)
    # soft membership probabilities if available
    probs = getattr(clusterer, 'probabilities_', None)
    if probs is None:
        probs = np.ones_like(labels, dtype=float)
        probs[labels < 0] = 0.0
    return labels, probs


def obb_from_points(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (center(3,), extent(3,), R(3,3), yaw_rad) from cluster points."""
    pcd = to_o3d(xyz)
    if len(pcd.points) < 3:
        c = xyz.mean(axis=0)
        return c, np.array([0.1, 0.1, 0.1]), np.eye(3), 0.0
    obb = pcd.get_oriented_bounding_box()
    c = np.asarray(obb.center)
    extent = np.asarray(obb.extent)
    R = np.asarray(obb.R)
    # choose x-axis of OBB as heading, project to XY for yaw
    x_axis = R[:, 0]
    yaw = math.atan2(x_axis[1], x_axis[0])
    return c, extent, R, yaw


# ------------------------------
# Pipeline per-frame
# ------------------------------

def process_frame(points: np.ndarray,
                  sem_labels: Optional[np.ndarray],
                  args: argparse.Namespace,
                  seq: str,
                  frame_idx: int,
                  out_dir: Path) -> FrameResult:
    # Prepare colors (reflectance)
    intens = points[:, 3]
    xyz = points[:, :3]

    # 1) Downsample (adaptive)
    if args.voxel > 0:
        # uniform voxel if set; otherwise adaptive
        pcd = to_o3d(xyz)
        xyz_ds = np.asarray(pcd.voxel_down_sample(args.voxel).points)
    else:
        xyz_ds = adaptive_range_voxel_downsample(points, bins=args.bins, voxel_sizes=args.voxels)

    # 2) Estimate normals (helps with RANSAC and region ops)
    pcd_ds = to_o3d(xyz_ds)
    estimate_normals_inplace(pcd_ds, radius=args.normal_radius, max_nn=args.normal_knn)

    # 3) Ground mask
    if args.use_labels and (sem_labels is not None):
        ground_mask = np.isin(sem_labels, args.ground_ids)
        # map ground to downsampled set via nearest neighbor for simplicity
        # (fast approximate: build KDTree on original xyz)
        tree = o3d.geometry.KDTreeFlann(to_o3d(xyz))
        ground_mask_ds = np.zeros(len(xyz_ds), dtype=bool)
        for i, p in enumerate(xyz_ds):
            _, idx, _ = tree.search_knn_vector_3d(p, 1)
            ground_mask_ds[i] = bool(ground_mask[idx[0]])
    else:
        if args.use_csf and HAS_CSF:
            ground_mask_ds = csf_ground_mask(xyz_ds, cloth_res=args.csf_res, rigidness=args.csf_rigid, slope_smooth=args.csf_smooth)
        else:
            ground_mask_ds = ransac_ground_mask(pcd_ds, distance_thresh=args.ransac_dist, max_planes=args.ransac_max_planes, min_inliers=args.ransac_min_inliers)

    # 4) Non-ground clustering
    xyz_obj = xyz_ds[~ground_mask_ds]
    labels, probs = hdbscan_cluster(xyz_obj, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, leaf=args.hdb_leaf)

    # 5) Boxes per cluster
    clusters: List[ClusterBox] = []
    uniq = sorted([int(u) for u in np.unique(labels) if u >= 0])
    for cid in uniq:
        mask = labels == cid
        if np.count_nonzero(mask) < args.min_points_box:
            continue
        pts = xyz_obj[mask]
        c, extent, R, yaw = obb_from_points(pts)
        clusters.append(ClusterBox(cluster_id=cid,
                                   n_points=int(pts.shape[0]),
                                   center=(float(c[0]), float(c[1]), float(c[2])),
                                   extent=(float(extent[0]), float(extent[1]), float(extent[2])),
                                   yaw=float(yaw),
                                   prob_mean=float(probs[mask].mean())))

    # 6) Save JSON
    res = FrameResult(sequence=str(seq),
                      frame_idx=int(frame_idx),
                      n_points_in=int(xyz.shape[0]),
                      n_points_proc=int(xyz_ds.shape[0]),
                      n_points_nonground=int(xyz_obj.shape[0]),
                      n_clusters=len(clusters),
                      clusters=clusters)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{frame_idx:06d}.json"
    with open(out_json, 'w') as f:
        json.dump({
            **{k: v for k, v in asdict(res).items() if k != 'clusters'},
            'clusters': [asdict(c) for c in clusters]
        }, f, indent=2)

    # 7) Optional visualization dump
    if args.save_vis:
        # colorize: gray ground, cluster colors for objects
        colors = np.zeros((xyz_ds.shape[0], 3), dtype=float) + 0.6
        colors[~ground_mask_ds] = 0.2  # default dark for non-ground
        # palette for clusters
        rng = np.random.default_rng(42)
        palette: Dict[int, np.ndarray] = {}
        for cid in uniq:
            palette[cid] = rng.random(3)
        # assign cluster colors
        obj_idx = np.where(~ground_mask_ds)[0]
        for local_i, idx in enumerate(obj_idx):
            cid = labels[local_i]
            if cid >= 0:
                colors[idx] = palette[cid]
        vis_pcd = to_o3d(xyz_ds)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out_dir / f"{frame_idx:06d}.ply"), vis_pcd)
        # save OBBs as LineSets
        line_sets = []
        for c in clusters:
            # build OBB geometry
            obb = o3d.geometry.OrientedBoundingBox(center=o3d.utility.Vector3dVector(np.array([c.center]).reshape(1,3))[0],
                                                   R=o3d.geometry.OrientedBoundingBox().R)  # placeholder
            # Recreate OBB from center/extent/yaw on Z
            cx, cy, cz = c.center
            dx, dy, dz = c.extent
            Rz = np.array([[math.cos(c.yaw), -math.sin(c.yaw), 0.0],
                           [math.sin(c.yaw),  math.cos(c.yaw), 0.0],
                           [0.0,              0.0,             1.0]])
            obb = o3d.geometry.OrientedBoundingBox(center=np.array([cx, cy, cz]), R=Rz, extent=np.array([dx, dy, dz]))
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            o3d.io.write_line_set(str(out_dir / f"{frame_idx:06d}_box_{c.cluster_id}.ply"), line_set)

    return res


# ------------------------------
# Main runner
# ------------------------------

def find_frames(args: argparse.Namespace) -> List[Tuple[np.ndarray, Optional[np.ndarray], str, int, Path]]:
    """Yield (points, sem_labels_or_None, seq, idx, out_dir_for_frame)."""
    outputs = []
    if args.bin:
        pts = read_kitti_bin(args.bin)
        outputs.append((pts, None, 'bin', 0, Path(args.out)))
        return outputs

    seq = args.sequence
    root = Path(args.dataset_root)
    seq_dir = root / 'sequences' / seq
    velo_dir = seq_dir / 'velodyne'
    label_dir = seq_dir / 'labels'
    if not velo_dir.exists():
        raise SystemExit(f"Not found: {velo_dir} (expected KITTI/SemanticKITTI layout)")

    start = int(args.start)
    end = int(args.end)
    frame_ids = list(range(start, end + 1))
    for fid in frame_ids:
        bin_path = velo_dir / f"{fid:06d}.bin"
        if not bin_path.exists():
            # skip silently so you can point to partial sets
            continue
        pts = read_kitti_bin(bin_path)
        sem: Optional[np.ndarray] = None
        if args.use_labels and (label_dir.exists()):
            lab_path = label_dir / f"{fid:06d}.label"
            if lab_path.exists():
                sem, _ = read_semantic_label(lab_path)
        out_dir = Path(args.out) / f"sequence{seq}"
        outputs.append((pts, sem, str(seq), fid, out_dir))
    return outputs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Million-point LiDAR pipeline (modern variant)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--dataset_root', type=str, help='Root of KITTI/SemanticKITTI dataset')
    src.add_argument('--bin', type=str, help='Single KITTI .bin file path')

    p.add_argument('--sequence', type=str, default='08', help='Sequence id (e.g., 00..21)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=20)

    # Downsampling
    p.add_argument('--voxel', type=float, default=0.0, help='Uniform voxel size (meters). If 0, use range-adaptive.')
    p.add_argument('--bins', type=float, nargs='+', default=[15.0, 30.0, 60.0, 120.0], help='Range bins for adaptive DS (meters)')
    p.add_argument('--voxels', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.4], help='Voxel sizes per bin (meters)')

    # Normals
    p.add_argument('--normal-radius', type=float, default=0.7)
    p.add_argument('--normal-knn', type=int, default=30)

    # Ground removal
    p.add_argument('--use-labels', dest='use_labels', action='store_true', help='Use SemanticKITTI .label if present')
    p.add_argument('--no-use-labels', dest='use_labels', action='store_false')
    p.set_defaults(use_labels=True)

    p.add_argument('--ground-ids', type=int, nargs='+', default=[40,44,48,49,60,72], help='Semantic ids treated as ground (raw ids)')

    p.add_argument('--use-csf', action='store_true', help='Use CSF cloth simulation ground filter if labels unavailable')
    p.add_argument('--csf-res', type=float, default=0.5)
    p.add_argument('--csf-rigid', type=int, default=3)
    p.add_argument('--csf-smooth', action='store_true')

    p.add_argument('--ransac-dist', type=float, default=0.2)
    p.add_argument('--ransac-max-planes', type=int, default=2)
    p.add_argument('--ransac-min-inliers', type=int, default=20000)

    # Clustering
    p.add_argument('--min-cluster-size', type=int, default=40)
    p.add_argument('--min-samples', type=int, default=None)
    p.add_argument('--hdb-leaf', action='store_true', help='Use leaf cluster selection (finer clusters)')
    p.add_argument('--min-points-box', type=int, default=25)

    # Output
    p.add_argument('--out', type=str, default='./outputs')
    p.add_argument('--save-vis', action='store_true', help='Save PLYs for clouds/boxes')

    args = p.parse_args(argv)
    # sanity
    if len(args.voxels) != len(args.bins):
        p.error("--voxels length must match --bins length")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    frames = find_frames(args)
    if len(frames) == 0:
        raise SystemExit("No frames found to process. Check paths or indices.")
    for pts, sem, seq, fid, out_dir in tqdm(frames, desc='Frames'):
        try:
            res = process_frame(pts, sem, args, seq, fid, out_dir)
        except Exception as e:
            print(f"[WARN] Failed frame {seq}/{fid:06d}: {e}", file=sys.stderr)
            continue
    print("Done.")


if __name__ == '__main__':
    main()

