# pointcloud-pipeline-modern

End-to-end LiDAR point-cloud pipeline that runs well on a laptop.
Pipeline: range-adaptive downsampling → robust ground (CSF by default; Patchwork++ optional) → HDBSCAN clustering → PCA OBBs → high-quality off-screen rendering.
Datasets: MulRan ParkingLot (Ouster, .bin identical to KITTI Velodyne format) and optional SemanticKITTI labels.


## Method
Range-adaptive downsampling (finer near, coarser far) for speed without losing shape.

Robust ground: Cloth Simulation Filter (CSF) by default; Patchwork++ is a fast, self-adaptive alternative for multi-level ground. 

HDBSCAN clustering (handles varying density; leaf mode produces small, homogeneous clusters vs eom). 

PCA-based oriented boxes (simple OBB via principal axes).

High-quality rendering: Open3D OffscreenRenderer with MSAA/TAA and ACES tone-mapping; headless/EGL friendly.

## Quickstart

#### 1) Install (Python)
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Open3D complains about GL on Linux:
```bash
sudo apt install -y libgl1 libegl1 libxext6 libsm6 libglib2.0-0
```

#### 2) Dataset layout
This repo expects KITTI-style folders:
```python
data/mulran_parkinglot/sequences/00/
  ├─ velodyne/000000.bin 000001.bin ...
  └─ calib.txt           # minimal line is fine for this pipeline
```

MulRan states LiDAR data is binary (same as KITTI), so standard KITTI readers work. If your files are timestamp-named, simply reindex them to 000000.bin, 000001.bin, ....

SemanticKITTI (optional): same KITTI layout, plus labels/*.label. Each label is a 32-bit uint: low 16 bits = semantic, high 16 bits = instance id (temporal).

## Run the pipeline (MulRan)
```bash
python Lidar_point_pipeline.py \
  --dataset_root ./data/mulran_parkinglot \
  --sequence 00 --start 0 --end 200 \
  --no-use-labels --use-csf \
  --bins 10 25 50 100 --voxels 0.05 0.08 0.15 0.30 \
  --min-cluster-size 50 --hdb-leaf \
  --out ./outputs/mulran_pl --save-vis
```

CSF = Cloth Simulation Filter (documented in PDAL). You can keep the simple plane fallback or wire PDAL’s CSF stage. 

Prefer HDBSCAN leaf selection when you want tighter per-object groupings; eom tends to pick larger, persistent clusters.


Output per frame lives in ./outputs/mulran_pl/sequence00/:

NNNNNN.ply – merged, colorized cloud

NNNNNN_box_K.ply – OBB line sets per cluster

optional JSON logs (parameters, timings)

## Render high-quality video (PNG → MP4)

Tip: render at 4K then downscale to 1080p for crisper results.

```bash
python render_sequence_hq.py \
  --in  ./outputs/mulran_pl/sequence00 \
  --out ./renders/mulran_seq00_frames_4k \
  --w 3840 --h 2160 \
  --stable-camera --orbit 0.25 \
  --point-size 3.0 --line-width 2.0 \
  --gain 1.3 --gamma 0.9 --accum 3
```

Encode with ffmpeg:

```bash
ffmpeg -y -framerate 15 -i ./renders/mulran_seq00_frames_4k/%06d.png \
  -vf "scale=1920:-2:flags=lanczos" \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p ./renders/mulran_seq00_1080p.mp4
```

# Why does this look good?

--stable-camera = one robust, quantile-fitted camera for all frames (no zoom jitter).

Bigger --point-size, MSAA/TAA, and ACES tone-mapping reduce speckling and lift midtones (Open3D OffscreenRenderer + ColorGrading). 

--gain/--gamma brighten intensity-colored clouds; --accum 3 overlays a few past frames to increase perceived density.


## Troubleshooting (tiny/dim render)

Stable camera: add --stable-camera; optionally clamp Z, e.g., --z -2 6.

Increase point size: try --point-size 3.0–4.0.

Color lift: --gain 1.2~1.5, --gamma 0.85~0.95.

Densify visually: --accum 3.

Render large, then downscale: 3840×2160 → 1920×1080 (Lanczos).

Headless/EGL: we use OffscreenRenderer.render_to_image() and look-at setup_camera(...). 

## Project structure

```python
├─ Lidar_point_pipeline.py        # pipeline (downsample → ground → cluster → OBB)
├─ render_sequence_hq.py          # high-quality offscreen renderer (PNG)
├─ cloth_nodes.txt                # CSF notes/config (optional)
├─ outputs/
│   └─ mulran_pl/sequence00/      # NNNNNN.ply, *_box_*.ply, logs
└─ renders/
    └─ mulran_seq00_frames_4k/    # %06d.png → MP4
```

## References & further reading

## References

- **MulRan dataset** — format notes (“LiDAR binary, same as KITTI”):  
  [Project page (Google Sites)](https://sites.google.com/view/mulran-dataset/home) •
  [Format details (ICRA’20 paper, §C Data Description and Format)](https://gisbi-kim.github.io/publications/gkim-2020-icra.pdf)  
  *The paper explicitly states the Ouster `.bin` files contain x,y,z,intensity and the format is identical to KITTI’s.* :contentReference[oaicite:0]{index=0}

- **SemanticKITTI** — KITTI layout, **32-bit labels** (low 16 bits semantic, high 16 bits instance):  
  [semantic-kitti.org/dataset.html](https://semantic-kitti.org/dataset.html) :contentReference[oaicite:1]{index=1}

- **Open3D** — Offscreen rendering & camera `setup_camera(...)`; ColorGrading **ACES** tone mapping:  
  [OffscreenRenderer docs](https://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.OffscreenRenderer.html) •
  [ColorGrading (tone mapping)](https://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.ColorGrading.html) :contentReference[oaicite:2]{index=2}

- **HDBSCAN** — cluster selection (`leaf` vs `eom`) and parameter guidance:  
  [hdbscan.readthedocs.io — Parameter selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) :contentReference[oaicite:3]{index=3}

- **CSF (Cloth Simulation Filter)** — PDAL stage:  
  [pdal.io — filters.csf](https://pdal.io/en/stable/stages/filters.csf.html) :contentReference[oaicite:4]{index=4}

- **Patchwork / Patchwork++** — fast, robust, self-adaptive ground segmentation:  
  [Patchwork (GitHub)](https://github.com/LimHyungTae/patchwork) •
  [Patchwork++ (GitHub)](https://github.com/url-kaist/patchwork-plusplus) :contentReference[oaicite:5]{index=5}
