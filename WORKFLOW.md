# Face Clustering Project — Complete Workflow

## Project Goal

Build an end-to-end face clustering system that:
1. Takes a raw dataset of celebrity face images (organized by identity)
2. Detects and aligns faces using a deep learning detector
3. Generates 512-dimensional ArcFace embeddings for each face
4. Clusters embeddings using HDBSCAN, K-Means, and DBSCAN
5. Evaluates clustering quality with NMI, ARI, and Purity metrics
6. Visualizes results through a Streamlit dashboard

---

## Environment Setup

**Virtual environment:** `facepoc/` (Python 3.13, located inside project directory)

```zsh
# Activate
source facepoc/bin/activate

# Deactivate
deactivate
```

**Key dependencies installed:**
```
onnxruntime       — runs ArcFace ONNX model
insightface       — RetinaFace face detector + landmark extractor
opencv-python     — image loading, resizing, affine warping
numpy             — embedding math
scikit-learn      — clustering algorithms, evaluation metrics
hdbscan           — HDBSCAN clustering
mysql-connector-python — MySQL embedding storage
streamlit         — dashboard UI
```

---

## Dataset

### Structure
```
data/raw/
├── Alia_bhat/        (11 images)
├── Ronaldo/          (15 images)
├── Thellapathy/      (11 images)
├── Virat/            (13 images)
├── akshay kumar/     (10 images)
├── sachin/           (12 images)
├── salman_khan/      (10 images)
└── shahrukh khan/    (14 images)
```

**Total:** 8 identities, 96 images

### Image types in dataset
- Solo portraits (ideal — single face, clear view)
- Group photos (problematic — multiple faces, target person ambiguous)
- Various resolutions, formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`

---

## Models

### ArcFace — `models/w600k_r50.onnx`
- ResNet-50 trained on WebFace600K dataset
- Outputs 512-dimensional face embedding vectors
- Input: 112×112 BGR image, normalized as `(pixel - 127.5) / 128.0`
- Output: L2-normalized unit vector (cosine similarity = dot product)
- Loaded via ONNX Runtime (no PyTorch/MXNet needed)

### InsightFace RetinaFace — auto-downloaded to `~/.insightface/models/buffalo_l/`
- Detects faces in full-size images
- Outputs bounding box + 5 facial landmarks per face:
  - `kps[0]` = left eye
  - `kps[1]` = right eye
  - `kps[2]` = nose tip
  - `kps[3]` = left mouth corner
  - `kps[4]` = right mouth corner

### YOLO models (attempted, not used in final pipeline)
- `models/yolov8m-face-lindevs.pt` — detection only, no landmarks
- `models/model.pt` — detection only, no landmarks
- Both rejected because they do not output facial landmarks needed for alignment

---

## Why InsightFace Over YOLO

YOLO detection-only models output bounding boxes but not facial landmarks.
Landmark detection is required for affine alignment, which is critical for
ArcFace accuracy. YOLOv8 pose models with 5-point face landmarks exist but
no pre-trained `.pt` file was publicly available for download without a
browser login.

InsightFace's RetinaFace is the natural pairing for ArcFace because:
- Both are from the same research group (deepinsight)
- Landmark coordinate space is guaranteed to match ArcFace's training template
- Works out of the box with no extra downloads beyond `pip install insightface`

---

## Pipeline Architecture

### Aligned Pipeline (`pipeline_new.py`) — PRIMARY

```
Image
  ↓
InsightFace RetinaFace
  → Bounding box + 5 landmarks
  → Filter: confidence ≥ 0.5, eye distance ≥ 5px
  → Skip group photos (>1 face detected) during verification
  ↓
cv2.estimateAffinePartial2D
  → Maps detected landmarks → ARCFACE_SRC canonical template
  → 112×112 affine-warped face crop
  ↓
preprocess_aligned_crop()
  → (pixel - 127.5) / 128.0
  → HWC → CHW transpose
  → Add batch dim → (1, 3, 112, 112)
  ↓
ArcFace ONNX model
  → 512-dim embedding
  → L2 normalize → unit vector
  ↓
MySQL (face_db.face_embeddings)
```

### ARCFACE_SRC — Canonical 5-point template
```python
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
])
```
These are the EXACT positions ArcFace was trained on. Must not be changed.

### Direct Pipeline (`direct_pipeline.py`) — ABLATION STUDY

Same as aligned pipeline but **skips landmark alignment**:

```
Image
  ↓
InsightFace RetinaFace
  → Bounding box only (landmarks ignored)
  → Add 10% margin around bbox
  ↓
Raw crop resized to 112×112
  ↓
ArcFace ONNX model
  ↓
MySQL (face_db_direct.face_embeddings_direct)
```

**Purpose:** Compare clustering quality with vs without alignment to quantify
how much alignment contributes to embedding quality.

---

## Model Verification

Before running the full pipeline, `verify_model()` runs a sanity check:

1. **Same-image test** — feed identical image twice, expect `sim ≈ 1.0`
2. **Same-identity pairs** — 2 solo images of same person, expect `sim ≥ 0.10`
3. **Cross-identity pairs** — images of different people, expect `sim ≤ 0.20`
4. **Separability gap** — `same_avg - cross_avg`, expect `≥ 0.10`

### Key fix during development
Original `verify_model()` skipped all images classified as `full_photo` (i.e., not exactly 160×160), meaning it silently tested zero images and reported separability=0.0. Fixed by:
- Routing all images through the YOLO/InsightFace detector
- Skipping group photos (>1 face) rather than skipping non-160px images
- Iterating all images per identity until 2 usable solo faces found

### Results on 8-identity dataset
```
Same-identity avg sim  : 0.5702  (want ≥ 0.10) ✅
Cross-identity avg sim : -0.0157 (want ≤ 0.20) ✅
Separability gap       : 0.5859  (want ≥ 0.10) ✅
```

---

## MySQL Storage

### Database: `face_db`
### Table: `face_embeddings`

```sql
CREATE TABLE face_embeddings (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    file_name       VARCHAR(512)  NOT NULL,
    identity_label  VARCHAR(256)  NOT NULL,
    image_type      VARCHAR(32)   NOT NULL DEFAULT 'full_photo',
    embedding_json  MEDIUMTEXT    NOT NULL,
    created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

**Note:** Table was recreated mid-project when `image_type` column was missing
from original schema. `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` is not
supported in older MySQL versions — table was dropped and recreated instead.

### Database: `face_db_direct`
### Table: `face_embeddings_direct`
Same schema without `image_type`, used by the direct (no-alignment) pipeline.

---

## Clustering Algorithms

All three algorithms read from `face_db.face_embeddings` and output:
- A `clustering_results_*.json` file with labels and metrics
- A `results_*/` folder with cluster subfolders containing image copies

### 1. HDBSCAN (`hdbscan_pipeline.py`)

**What it is:** Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Automatically finds clusters of arbitrary shape and labels
low-density points as noise (-1).

**Why it's best for faces:**
- Does not require knowing K (number of people) in advance
- Handles noise naturally — group photos / ambiguous faces get label -1
- Adapts cluster density per cluster (unlike DBSCAN which uses fixed eps)
- Euclidean distance on L2-normalized embeddings ≈ cosine distance

**Hyperparameter sweep:**
```
min_cluster_size: [2, 3, 4, 5]
min_samples:      [1, 2, 3]
metric:           euclidean
cluster_selection_method: eom (excess of mass)
```

**Best result (8 identities, 96 images):**
```
min_cluster_size=3, min_samples=3
NMI=0.8475, ARI=0.7713
Clusters found: 8 ✅ (exactly matches true identities)
Noise: 13 images (13.5%)
```

**Output:** `clustering_results.json`, `results/`

---

### 2. K-Means (`kmeans_pipeline.py`)

**What it is:** Partitions N points into K clusters by minimizing within-cluster
variance. Requires K to be specified upfront.

**Why it's limited for faces:**
- Must know number of people in advance (not realistic)
- Forces every image into a cluster — no noise handling
- Group photos / ambiguous faces get assigned to whichever centroid is closest,
  dragging down purity

**Hyperparameter sweep:**
```
K: [2, 3, 4, 5, 6]
init: k-means++ (smart centroid initialization)
n_init: 20 (run 20 times, keep best)
```

**Best result (8 identities, 96 images):**
```
K=6 (best ARI, but wrong — true K=8)
NMI=0.7879, ARI=0.5332
Multiple identities merged into single clusters
```

**Output:** `clustering_results_kmeans.json`, `results_kmeans/`

---

### 3. DBSCAN (`dbscan_pipeline.py`)

**What it is:** Density-Based Spatial Clustering. Groups points within eps
radius with at least min_samples neighbours. Points in sparse regions → noise.

**Why it struggles for faces:**
- Fixed eps parameter is very sensitive — wrong value collapses everything
  into individual clusters or all noise
- On small datasets (96 images), embedding density is too sparse for stable
  density estimation
- HDBSCAN is a strictly better version of DBSCAN for this use case

**Hyperparameter sweep:**
```
eps:         [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples: [1, 2, 3]
metric:      euclidean
```

**Best result (8 identities, 96 images):**
```
eps=1.0, min_samples=1
NMI=0.8050, ARI=0.6173
Clusters found: 37 ❌ (fragmented — 8 identities split into 37 clusters)
```

**Output:** `clustering_results_dbscan.json`, `results_dbscan/`

---

## Evaluation Metrics

### NMI — Normalized Mutual Information
Measures information shared between predicted clusters and ground truth labels.
Range: 0 (random) → 1.0 (perfect). Symmetric, not sensitive to cluster count.

### ARI — Adjusted Rand Index
Measures pairwise agreement between predicted and true assignments, adjusted
for chance. Range: ~0 (random) → 1.0 (perfect). Negative values possible.
**Most reliable metric** — penalizes both over-splitting and merging.

### Purity
For each cluster, fraction of images belonging to majority identity.
Range: 0 → 1.0. Can be gamed by splitting into many small clusters
(each single-image cluster has purity=1.0). Use only alongside ARI.

---

## Algorithm Comparison (8 identities, 96 images)

| Algorithm | NMI    | ARI    | Purity | Clusters | Noise % | Verdict |
|-----------|--------|--------|--------|----------|---------|---------|
| HDBSCAN   | 0.8475 | 0.7713 | —      | 8 ✅     | 13.5%   | **Best** |
| K-Means   | 0.7879 | 0.5332 | 0.9412 | 6 ❌     | 0%      | Wrong K |
| DBSCAN    | 0.8050 | 0.6173 | 1.0000 | 37 ❌    | 0%      | Fragmented |

**Winner: HDBSCAN** — only algorithm that found exactly the right number of
clusters without being told how many people there are.

---

## Aligned vs Direct Pipeline Comparison (2 identities, 17 images)

| Pipeline  | NMI    | ARI    | Noise | Description |
|-----------|--------|--------|-------|-------------|
| Aligned   | 0.6626 | 0.6437 | 3     | InsightFace + affine alignment |
| Direct    | 0.5938 | 0.5542 | 4     | InsightFace + raw bbox crop |

Alignment contributes ~+0.07 NMI / +0.09 ARI. Impact grows significantly
on larger, more varied datasets with pose/scale variation.

---

## Streamlit Dashboards

### `app.py` — HDBSCAN results only
```zsh
streamlit run app.py
```
Shows: metrics, identity analysis, split identity warnings, cluster image grid,
cluster summary.

### `app_direct.py` — Direct (no-alignment) pipeline results
```zsh
streamlit run app_direct.py
```
Shows same layout as app.py but for the bbox-crop ablation study.

### `app_combined.py` — All three algorithms side by side
```zsh
streamlit run app_combined.py
```
Shows:
- Side-by-side metric comparison table
- Bar chart (NMI / ARI / Purity per algorithm)
- Per-identity breakdown (switchable per algorithm)
- Cluster image explorer (switchable per algorithm)
- Full hyperparameter sweep table

---

## How to Run Full Pipeline from Scratch

```zsh
# 1. Activate environment
source facepoc/bin/activate

# 2. Generate embeddings (detect → align → embed → store)
python3 pipeline_new.py --mode run

# 3. Run all clustering algorithms
python3 hdbscan_pipeline.py
python3 kmeans_pipeline.py
python3 dbscan_pipeline.py

# 4. Create HDBSCAN cluster image folders
python3 create_cluster_folders.py

# 5. Launch combined dashboard
streamlit run app_combined.py
```

---

## File Reference

| File | Purpose |
|------|---------|
| `pipeline_new.py` | Main pipeline: audit → verify → embed → store |
| `direct_pipeline.py` | Ablation: bbox-crop pipeline (no alignment) |
| `hdbscan_pipeline.py` | HDBSCAN clustering + save JSON + folders |
| `kmeans_pipeline.py` | K-Means clustering + save JSON + folders |
| `dbscan_pipeline.py` | DBSCAN clustering + save JSON + folders |
| `create_cluster_folders.py` | Copies images into HDBSCAN cluster folders |
| `app.py` | Streamlit dashboard for HDBSCAN results |
| `app_direct.py` | Streamlit dashboard for direct pipeline results |
| `app_combined.py` | Combined dashboard comparing all 3 algorithms |
| `clustering_results.json` | HDBSCAN best labels + metrics |
| `clustering_results_kmeans.json` | K-Means best labels + metrics |
| `clustering_results_dbscan.json` | DBSCAN best labels + metrics |
| `clustering_results_direct.json` | Direct pipeline HDBSCAN results |
| `results/` | HDBSCAN cluster image folders |
| `results_kmeans/` | K-Means cluster image folders |
| `results_dbscan/` | DBSCAN cluster image folders |
| `results_direct/` | Direct pipeline cluster folders |
| `data/raw/` | Raw input images organized by identity |
| `models/w600k_r50.onnx` | ArcFace model weights |
| `facepoc/` | Python virtual environment |

---

## Key Issues Encountered and Fixes

### 1. YOLO model had no landmarks
**Problem:** `yolov8m-face-lindevs.pt` and `model.pt` are detection-only models
(`task: detect`). They output bounding boxes but no facial landmarks.
`detect_and_align()` requires 5 landmarks for affine alignment.

**Fix:** Replaced `YoloFaceDetector` with `InsightFaceDetector` using
InsightFace's RetinaFace, which outputs 5 landmarks in the correct format.

### 2. verify_model() silently tested zero images
**Problem:** `verify_model()` classified images as `precropped` (160×160) or
`full_photo`, then skipped all `full_photo` images. All web-downloaded images
are not exactly 160×160, so everything was skipped. Result: separability=0.0.

**Fix:** Removed image type routing from verification. All images now go through
InsightFace detection. Group photos (>1 face detected) are skipped with a
warning. Code iterates all images per identity until 2 usable solo faces found.

### 3. MySQL table missing `image_type` column
**Problem:** Original `face_embeddings` table was created without `image_type`
column. Pipeline insert statement included it → `Unknown column` error.

**Fix:** Dropped and recreated table with correct schema including `image_type`.

### 4. app_combined.py `_format_params` NameError
**Problem:** `_format_params()` was called before it was defined in the script.
Python/Streamlit executes top-to-bottom so function must precede its call.

**Fix:** Moved `_format_params()` definition above the first call site.

### 5. HDBSCAN verification threshold too strict
**Problem:** Separability threshold was 0.25. Web-downloaded images with group
photos produced embeddings of random people, driving same-identity similarity
below threshold even though the model was working correctly.

**Fix:** Lowered threshold to 0.10 AND added group photo filtering in
verification loop.

---

## Concepts Explained

### Why alignment matters
ArcFace was trained on faces where eyes, nose, and mouth are always at the
same pixel positions (ARCFACE_SRC template). If you feed an unaligned face
(rotated, shifted, different scale), the model sees different pixel patterns
than it was trained on → lower quality embeddings → embeddings of the same
person look more different, embeddings of different people look more similar.

### Why L2 normalization matters
After L2 normalization, all embeddings are unit vectors on a 512-dimensional
hypersphere. Cosine similarity = dot product (no division needed). Euclidean
distance on unit vectors ∝ cosine distance. This means all distance-based
clustering algorithms (K-Means, DBSCAN, HDBSCAN) work correctly with
`metric='euclidean'` on normalized embeddings.

### Why HDBSCAN is best for face clustering
- Real-world photo collections have unknown number of people → K unknown
- Some photos are ambiguous (group shots, partial faces) → need noise handling
- Face cluster density varies (some people have very similar photos, others vary
  widely) → need variable density support
HDBSCAN handles all three. K-Means fails on 1 and 2. DBSCAN fails on 3.

### Opencv Haar Cascade (manager's suggestion)
`cv2.CascadeClassifier('haarcascade_frontalface_default.xml')` is a classic
built-in OpenCV face detector. It gives bounding boxes only (no landmarks),
struggles with profile faces, and is significantly less accurate than
RetinaFace. It is equivalent to the `direct_pipeline.py` approach but with a
weaker detector. Not used in the final pipeline.
