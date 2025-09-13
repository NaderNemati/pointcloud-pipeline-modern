# Create a tiny SemanticKITTI-like toy dataset with a couple of frames.
# It will include the folder structure and a few synthetic KITTI-style .bin scans (float32 [x,y,z,intensity])
# plus matching .label files (uint32 with low 16 bits semantic, high 16 bits instance).
import os, struct, zipfile, numpy as np, io, json, math, random, pathlib

root = pathlib.Path("/mnt/data/semantickitti_toy")
seq_dir = root / "sequences" / "08"
velodyne_dir = seq_dir / "velodyne"
labels_dir = seq_dir / "labels"
for d in [velodyne_dir, labels_dir]:
    d.mkdir(parents=True, exist_ok=True)

def make_toy_scan(n_ground=20000, n_objs=3, obj_pts=4000, seed=42):
    rng = np.random.default_rng(seed)
    # Ground plane: uniform on a disk with slight noise in z
    r = np.sqrt(rng.random(n_ground)) * 40.0
    theta = rng.random(n_ground) * 2*np.pi
    xg = r * np.cos(theta)
    yg = r * np.sin(theta)
    zg = rng.normal(0.0, 0.05, n_ground)
    ig = rng.random(n_ground).astype(np.float32) * 1.0

    pts = [np.stack([xg, yg, zg, ig], axis=1)]
    sem = [np.full(n_ground, 40, dtype=np.uint16)]  # 40 = road

    # A few "objects": cylinders standing on ground
    for k in range(n_objs):
        cx, cy = rng.uniform(-20,20), rng.uniform(-20,20)
        h = rng.uniform(1.5, 2.5)
        rad = rng.uniform(0.5, 1.2)
        th = rng.random(obj_pts) * 2*np.pi
        rr = rad + rng.normal(0, 0.05, obj_pts)
        x = cx + rr*np.cos(th)
        y = cy + rr*np.sin(th)
        z = rng.random(obj_pts) * h
        i = rng.random(obj_pts).astype(np.float32)
        pts.append(np.stack([x,y,z,i], axis=1))
        sem.append(np.full(obj_pts, 10, dtype=np.uint16))  # 10 = car (just for demo)

    P = np.concatenate(pts, axis=0).astype(np.float32)
    sem = np.concatenate(sem, axis=0)
    # Shuffle
    idx = np.arange(P.shape[0])
    rng.shuffle(idx)
    P = P[idx]
    sem = sem[idx]
    # Build labels uint32: (instance<<16) | semantic
    inst = np.zeros_like(sem, dtype=np.uint16)
    labels = (inst.astype(np.uint32) << 16) | sem.astype(np.uint32)
    return P, labels

# Create two frames
frames = [0, 1]
for f in frames:
    P, L = make_toy_scan(seed=42+f)
    # write .bin
    bin_path = velodyne_dir / f"{f:06d}.bin"
    P.astype(np.float32).tofile(bin_path)
    # write .label
    label_path = labels_dir / f"{f:06d}.label"
    L.astype(np.uint32).tofile(label_path)

# Add a minimal calib file (not used by our script but common in KITTI layout)
calib_text = "P0: 1 0 0 0  0 1 0 0  0 0 1 0\n"
(seq_dir / "calib.txt").write_text(calib_text)

# Zip it for easy download
zip_path = "/mnt/data/semantickitti_toy_08.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob("*"):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(root)))

zip_path

