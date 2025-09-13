# pointcloud-pipeline-modern




## Method. 

We build an end-to-end LiDAR pipeline that is laptop-friendly and modernized for real scenes. We use range-adaptive downsampling (finer near, coarser far), robust ground segmentation (Cloth Simulation Filter by default; optional Patchwork++ for multi-level ground), HDBSCAN clustering (better than a single-ε DBSCAN on varying densities), and PCA-based oriented boxes. For visuals, we render off-screen with Open3D using MSAA/TAA and ACES tone-mapping to produce high-quality MP4/GIFs. Dataset-wise, we run on MulRan ParkingLot (Ouster LiDAR, .bin same as KITTI) and optionally SemanticKITTI if you want labels.










cd ~/Desktop/LiDar_Million_point

# 1) .gitignore (keeps outputs & venv out of git)
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
.venv/
.env

# Data & outputs (large/artifacts)
outputs/
renders/
data/
*.mp4
*.gif
*.png

# OS/editor
.DS_Store
EOF

# 2) requirements (what users need to run your scripts)
cat > requirements.txt << 'EOF'
numpy>=1.24
open3d>=0.18
hdbscan>=0.8.35
scikit-learn>=1.3
tqdm>=4.66
pyyaml>=6.0
Pillow>=10.0
# optional (if you want to drive CSF via PDAL routes later)
pdal>=3.1 ; platform_system!="Windows"
EOF

# 3) README (starter, edit author lines)
cat > README.md << 'EOF'
# pointcloud-pipeline-modern

End-to-end LiDAR point-cloud pipeline with modern tweaks — runs on a laptop.

**Highlights**
- Range-adaptive downsampling
- Robust ground: CSF by default; optional **Patchwork++** (fast, self-adaptive)
- Clustering: **HDBSCAN** (handles varying densities)
- 3D boxes: **PCA OBB**
- High-quality rendering (Open3D OffscreenRenderer + MSAA/TAA + ACES)

**Datasets**
- **MulRan / ParkingLot** – Ouster LiDAR `.bin` **same as KITTI** → works with KITTI loaders.  
- **SemanticKITTI** – Optional labels (32-bit packed: lower 16 bits semantic, upper 16 instance).

## Quickstart

Create an environment and install:
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

```




## Run the pipeline on MulRan
```bash
python Lidar_point_pipeline.py \
  --dataset_root /home/nader/Datasets/mulran_parkinglot \
  --sequence 00 --start 0 --end 200 \
  --no-use-labels --use-csf \
  --bins 10 25 50 100 --voxels 0.05 0.08 0.15 0.30 \
  --min-cluster-size 50 --hdb-leaf \
  --out ./outputs/mulran_pl --save-vis
```

## Render a high-quality video (PNG frames → MP4)
```bash
python render_sequence_hq.py \
  --in  ./outputs/mulran_pl/sequence00 \
  --out ./renders/mulran_seq00_frames_4k \
  --w 3840 --h 2160 --stable-camera --orbit 0.25 \
  --point-size 3.0 --line-width 2.0 --gain 1.3 --gamma 0.9 --accum 3
```

```bash
ffmpeg -y -framerate 15 -i ./renders/mulran_seq00_frames_4k/%06d.png \
  -vf "scale=1920:-2:flags=lanczos" \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p ./renders/mulran_seq00_1080p.mp4
```


## Method

Range-adaptive downsample → robust ground (CSF / Patchwork++) → HDBSCAN clustering (BEV) → PCA OBB → HQ off-screen rendering.

## References

MulRan: LiDAR binary format same as KITTI.

SemanticKITTI: 32-bit labels (low 16 bits = semantic; high 16 = instance).

Open3D OffscreenRenderer & camera setup.

HDBSCAN cluster selection (leaf vs eom).

PDAL CSF (Cloth Simulation Filter).

Patchwork / Patchwork++ for robust ground.

## MIT license

cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
EOF

## Initialize git and make the first commit

git init
git add README.md requirements.txt .gitignore LICENSE Lidar_point_pipeline.py render_sequence_hq.py cloth_nodes.txt
git commit -m "Initial commit: modern LiDAR pipeline (MulRan/KITTI) + HQ renderer"

