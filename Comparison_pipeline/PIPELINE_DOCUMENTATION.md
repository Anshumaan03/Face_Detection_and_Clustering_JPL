# Face Embedding Comparison Pipeline

**Author:** Anshumaan Singh Rathore  
**Models:** ArcFace · dlib-ResNet · FaceNet (VGGFace2) · SigLIP2  
**Detector:** InsightFace (buffalo_l / RetinaFace)  
**Clustering:** HDBSCAN (identical hyperparameters across all models)

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Architecture](#why-this-architecture)
3. [Project Structure](#project-structure)
4. [Module Reference](#module-reference)
   - [config.py](#configpy)
   - [common.py](#commonpy)
   - [embeddings.py](#embeddingspy)
   - [storage.py](#storagepy)
   - [clustering.py](#clusteringpy)
   - [run_pipeline.py](#run_pipelinepy)
   - [app.py](#apppy)
5. [Complete Execution Flow](#complete-execution-flow)
6. [Detection and Alignment — Per Model](#detection-and-alignment--per-model)
7. [Database Schema](#database-schema)
8. [Clustering Methodology](#clustering-methodology)
9. [Results and Metrics](#results-and-metrics)
10. [Setup and Running](#setup-and-running)
11. [Known Issues and Fixes Applied](#known-issues-and-fixes-applied)

---

## Overview

This pipeline compares the face clustering quality of four embedding models under strictly controlled conditions:

- **One shared detector** (InsightFace RetinaFace via `buffalo_l`) runs once per image, producing bounding box coordinates and 5-point facial landmarks.
- **Per-model alignment** branches apply each model's specific crop/warp convention before feeding into that model's extractor — the part that was broken in the original pipeline and caused ArcFace's ARI to drop from 0.777 → 0.28.
- **Identical HDBSCAN hyperparameters** across all 4 models ensure the clustering step does not introduce any bias into the comparison.
- **All embeddings are L2-normalized** centrally, after extraction, identically for every model.
- **MySQL** stores bounding boxes, landmarks, and embedding vectors persistently so detection and extraction do not need to be rerun each time clustering parameters are adjusted.
- **Streamlit** provides an interactive dashboard for metrics comparison and cluster folder browsing.

---

## Why This Architecture

### The Root Cause of the Original 0.777 → 0.28 ARI Drop

ArcFace (and most metric-learning face models) are trained on tightly aligned, landmark-warped crops — specifically 112×112 images where 5 facial landmark points (left eye, right eye, nose tip, left mouth corner, right mouth corner) are warped to a canonical template. The embedding space is only meaningful when the model is given input that matches this training distribution.

When the new pipeline fed raw/unaligned crops from the dataset loader directly to ArcFace — skipping InsightFace detection and alignment — the model was operating far outside its training distribution. This does not cause an error or warning; it silently produces embeddings that are numerically valid but semantically much weaker, which is exactly why the ARI dropped so dramatically.

**Fix:** run InsightFace detection on every image first to obtain both the bounding box and 5-point landmarks, then apply each model's specific alignment convention before passing the crop to that model's extractor.

### Why Per-Model Alignment Matters

Forcing all four models through the same alignment convention (e.g. ArcFace's norm_crop for everyone) is not a fair comparison — it would silently handicap any model that was not trained on that alignment. The three alignment strategies used are:

| Strategy | Models | Description |
|---|---|---|
| `insightface_norm_crop` | ArcFace | Similarity transform warp to ArcFace's 112×112 canonical 5-point template |
| `dlib_chip` | dlib-ResNet | Re-runs dlib's own shape predictor inside the InsightFace bbox; calls `dlib.get_face_chip()` |
| `bbox_margin_crop` | FaceNet, SigLIP2 | Square bbox crop with generous margin, resized to target resolution — no landmark warping |

dlib-ResNet requires its own shape predictor because `dlib.get_face_chip()` only accepts `dlib.full_object_detection` objects produced by dlib's own predictor — it cannot consume InsightFace's landmark format directly. InsightFace's bbox is still used to tell dlib's predictor where to look in the image, avoiding a second full-image detection pass.

FaceNet and SigLIP2 receive bbox-margin crops (no landmark warp) because neither model was trained on landmark-warped input: FaceNet (VGGFace2) expects a generous bbox crop, and SigLIP2 is a general vision-language model trained on natural images where tight warping would be counterproductive.

---

## Project Structure

```
face_pipeline/
├── config.py                  # All paths, model specs, DB config, HDBSCAN params
├── common.py                  # Image loading, detection (InsightFace), alignment (per-model)
├── embeddings.py              # One extractor class per model + L2 normalization
├── storage.py                 # MySQL schema creation, insert/load helpers
├── clustering.py              # HDBSCAN, metrics, cluster-folder export
├── run_pipeline.py            # Orchestration entry point
├── app.py                     # Streamlit visualization dashboard
├── requirements.txt           # Python dependencies
└── PIPELINE_DOCUMENTATION.md # This file
```

---

## Module Reference

### `config.py`

Central configuration — the only file that should need editing when adapting the pipeline to a new machine or dataset.

**Key settings:**

```python
DATA_ROOT = "data/raw"              # data/raw/<identity>/<images>
CLUSTERS_ROOT = "outputs/clusters"  # written by export_cluster_folders()

ARCFACE_ONNX_PATH = "..."           # path to arcface.onnx
DLIB_RESNET_PATH = "..."            # path to dlib_face_recognition_resnet_model_v1.dat
DLIB_SHAPE_PREDICTOR_PATH = "..."   # path to shape_predictor_68_face_landmarks.dat
FACENET_VGGFACE2_PATH = "..."       # path to facenet_vggface2.pt
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-224"  # HF model name

INSIGHTFACE_DET_MODEL = "buffalo_l" # detector model pack
DETECTOR_CTX_ID = -1                # -1 = CPU (macOS); 0 = first CUDA GPU (Linux/Windows)
```

**Per-model input specs** (the alignment branch selector):

```python
MODEL_INPUT_SPECS = {
    "arcface":     {"size": 112, "alignment": "insightface_norm_crop"},
    "dlib_resnet": {"size": 150, "alignment": "dlib_chip"},
    "facenet":     {"size": 160, "alignment": "bbox_margin_crop", "margin": 0.25},
    "siglip2":     {"size": 224, "alignment": "bbox_margin_crop", "margin": 0.40},
}
```

**HDBSCAN params** (identical for all models — not tuned per-model):

```python
HDBSCAN_PARAMS = {
    "min_cluster_size": 5,
    "min_samples": 3,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}
```

---

### `common.py`

Provides three responsibilities: image loading, face detection, and alignment.

**`load_image(path)`**  
Reads an image as BGR uint8 via `cv2.imread()`. Raises `FileNotFoundError` immediately on a bad file rather than returning `None` — prevents downstream errors from silently propagating.

**`iter_dataset(data_root)`**  
Generator that walks `data/raw/`, yielding `(identity, image_path)` tuples in sorted order. Works directly with the folder structure `data/raw/<person_name>/<images>`.

**`detect_faces(img_bgr, image_path="")`**  
Lazy-initializes a singleton `InsightFace.FaceAnalysis` instance (loaded once per process, reused for every image). Runs RetinaFace detection on the full image and returns a list of `FaceDetection` dataclass objects sorted by detection score, each containing:
- `bbox`: `np.ndarray` of shape `(4,)` — `[x1, y1, x2, y2]`
- `landmarks`: `np.ndarray` of shape `(5, 2)` — 5-point facial landmarks in InsightFace's canonical order (left eye, right eye, nose, left mouth, right mouth)
- `det_score`: float confidence
- `image_path`: for logging/debugging

**`align_face(img_bgr, face, model)`**  
Single entry point for alignment. Reads `config.MODEL_INPUT_SPECS[model]` to determine strategy, then dispatches to one of:

- `_similarity_transform_align()` — estimates a similarity transform (rotation + uniform scale + translation, no shear) that maps `face.landmarks` to `_ARCFACE_TEMPLATE_112` (the standard 112×112 5-point canonical face template). Uses `cv2.estimateAffinePartial2D` + `cv2.warpAffine`.
- `_dlib_chip_align()` — runs `dlib.shape_predictor` on the image within the InsightFace bbox to get dlib-native landmarks, then calls `dlib.get_face_chip()`. Note: 5-point and 68-point predictor files both work here since `get_face_chip` is point-count agnostic.
- `_bbox_margin_crop()` — computes a square crop centered on the bbox, expanded by `margin_frac` on each side, clips to image bounds, and resizes to `out_size`. No rotation or warping.

All alignment functions return a BGR uint8 numpy array at the model's expected resolution. Model-specific normalization (channel order, mean/std scaling) is handled in `embeddings.py`, not here.

---

### `embeddings.py`

One extractor class per model, all behind a common `.extract(aligned_crop) -> np.ndarray` interface.

**`ArcFaceExtractor`**  
Loads `arcface.onnx` via `onnxruntime`. Provider list checks for CoreML (macOS) before CUDA, with CPU as guaranteed fallback — avoids the "CUDAExecutionProvider not available" warning on Mac and actually accelerates on Apple Neural Engine.

Preprocessing inside `extract()`: converts BGR → RGB, scales pixel values to `[-1, 1]` via `(img - 127.5) / 128.0`, transposes HWC → CHW, adds batch dimension. This matches ArcFace ONNX models exported from InsightFace's training setup.

**`DlibResnetExtractor`**  
Loads `dlib_face_recognition_resnet_model_v1.dat` via `dlib.face_recognition_model_v1`. Calls `compute_face_descriptor(img_rgb)` on the already-aligned chip (BGR → RGB conversion inside `extract()`). Returns a 128-dimensional descriptor.

**`FaceNetExtractor`**  
Loads `facenet_vggface2.pt` into `facenet_pytorch.InceptionResnetV1(classify=False)`. The checkpoint includes `logits.weight`/`logits.bias` keys (the classification head from VGGFace2 training), which are stripped before `load_state_dict()` since we want only the 512-dim embedding trunk with `classify=False`.

Preprocessing: BGR → RGB, `(img - 127.5) / 128.0`, CHW transpose, batch dimension added. Same normalization as ArcFace since both were trained on similar face-image distributions.

**`SigLIP2Extractor`**  
Loads `google/siglip2-base-patch16-224` via `transformers.AutoModel` and `AutoImageProcessor`. The processor handles all normalization internally — the `extract()` method only needs to convert BGR → RGB before passing to the processor.

Calls `model.get_image_features(**inputs)`. In newer `transformers` versions (including the one on this machine), this returns a `BaseModelOutputWithPooling` wrapper object rather than a raw tensor — the extractor checks for `.pooler_output` and unwraps it when present, making the code robust across transformer version changes.

**`get_normalized_embedding(extractor, aligned_crop)`**  
The only path embeddings should flow through. Calls `extractor.extract()` then `l2_normalize()`. Centralizing normalization here (rather than inside each extractor) ensures it applies identically regardless of model — mixing normalized and raw vectors across models is a common silent-degradation source.

---

### `storage.py`

MySQL persistence layer. Creates and manages three tables:

**`faces`** — one row per detected face (per image):

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | auto-increment, used as foreign key in `embeddings` |
| `identity` | VARCHAR(255) | folder name from `data/raw/<identity>/` |
| `image_path` | VARCHAR(1024) UNIQUE | full path; UNIQUE prevents duplicate rows on re-run |
| `bbox` | JSON | `[x1, y1, x2, y2]` |
| `landmarks` | JSON | `[[x,y], ...]` 5 points |
| `det_score` | FLOAT | InsightFace detection confidence |

**`embeddings`** — four rows per face (one per model):

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | auto-increment |
| `face_id` | INT FK | references `faces.id` |
| `model` | VARCHAR(64) | `'arcface'`, `'dlib_resnet'`, `'facenet'`, `'siglip2'` |
| `dim` | INT | embedding dimensionality |
| `vector` | JSON | L2-normalized float array |
| UNIQUE | `(face_id, model)` | one embedding per (face, model) pair |

**`cluster_results`** — one row per (face, model, run_label):

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | auto-increment |
| `model` | VARCHAR(64) | model name |
| `run_label` | VARCHAR(255) | experiment label (default: `'default'`) |
| `face_id` | INT FK | references `faces.id` |
| `cluster_label` | INT | HDBSCAN output; `-1` = noise |

All inserts use `ON DUPLICATE KEY UPDATE`, so re-running the pipeline on the same dataset safely overwrites existing rows without creating duplicates.

`_ensure_database_exists()` connects without a database selected and runs `CREATE DATABASE IF NOT EXISTS` — no manual DB creation needed before the first run.

`load_embeddings_df(model)` joins `embeddings` and `faces`, parses JSON vectors back to `np.ndarray`, and returns a pandas DataFrame used directly by the clustering step.

---

### `clustering.py`

**`run_hdbscan(embeddings, params)`**  
Accepts an `(N, D)` L2-normalized embedding matrix and runs `hdbscan.HDBSCAN(**params).fit_predict()`. Returns a `(N,)` label array where `-1` indicates noise points HDBSCAN was not confident enough to assign to any cluster.

HDBSCAN was chosen over k-means because it does not require the number of clusters to be specified in advance, handles non-convex cluster shapes naturally, and produces a meaningful noise concept (-1 label) that exposes poor-quality embeddings rather than forcing every point into some cluster. With k-means, a bad embedding model would still produce `k` clusters at full coverage — hiding the quality signal. HDBSCAN's noise fraction is itself a meaningful metric in this comparison.

**`compute_metrics(true_labels, predicted_labels)`**  
Computes standard external clustering metrics, including noise points (`-1`) as their own label class rather than dropping them — silently dropping noise points would inflate scores for models with high noise fractions.

Metrics computed:
- **ARI (Adjusted Rand Index)**: measures agreement between predicted and true clustering, adjusted for chance. Range [-1, 1]; 1 = perfect, 0 = random.
- **NMI (Normalized Mutual Information)**: measures shared information between clusterings, normalized to [0, 1].
- **Homogeneity**: each predicted cluster contains only members of a single true identity.
- **Completeness**: all members of a true identity are assigned to the same predicted cluster.
- **V-measure**: harmonic mean of homogeneity and completeness (= NMI here since both are normalized the same way).
- **n_predicted_clusters**: how many clusters HDBSCAN found (excluding noise).
- **noise_fraction**: fraction of faces assigned `-1` (no confident cluster).

**`export_cluster_folders(df, model, run_label)`**  
Creates `outputs/clusters/<run_label>/<model>/cluster_<N>/` directories and symlinks (or copies, if symlinks fail) each image into the folder matching its predicted cluster. Noise points go into a `noise/` subfolder. The `identity__filename` naming convention inside each cluster folder enables quick visual purity inspection — if a cluster is pure, every filename will share the same identity prefix.

---

### `run_pipeline.py`

Orchestration entry point. Two phases:

**`run_extraction(limit)`**: iterates the dataset, runs detection once per image, runs alignment + extraction + storage once per model per image. The `tqdm` progress bar gives live feedback. Images where no face is detected are skipped and counted in the final log summary.

**`run_clustering_and_report(run_label)`**: calls `cluster_all_models()` (HDBSCAN for all 4 models), then `metrics_summary_table()` to print the comparison table.

CLI flags:
- `--limit N`: process only the first N images (useful for testing before a full run)
- `--skip-extraction`: skip detection/embedding, re-run only the clustering step (useful when only HDBSCAN parameters have changed)
- `--run-label LABEL`: tag this clustering run with a custom label (allows multiple runs to coexist in the database and cluster folders)

---

### `app.py`

Streamlit dashboard with three sections:

**Metrics comparison table**: the full per-model ARI/NMI/homogeneity/completeness/v-measure/noise-fraction table, formatted as an interactive dataframe.

**ARI bar chart**: `st.bar_chart(table["ari"])` — a quick visual ranking of models by clustering quality.

**Cluster browser**: a model selector + cluster selector that displays the face images in that cluster as a thumbnail grid. Automatically flags whether a cluster is "pure" (all images share one identity prefix in their filename) or "mixed" (multiple identities landed in the same cluster), based on the `identity__filename` naming scheme written by `export_cluster_folders()`.

Run with: `streamlit run app.py`

---

## Complete Execution Flow

```
python run_pipeline.py [--limit 50] [--skip-extraction] [--run-label default]
│
├─ Storage()
│    └─ _ensure_database_exists()    CREATE DATABASE IF NOT EXISTS face_db_new
│    └─ create_schema()              CREATE TABLE IF NOT EXISTS faces, embeddings, cluster_results
│
├─ load_all_extractors()             load all 4 model weights into memory (once)
│    ├─ ArcFaceExtractor()           onnxruntime session, arcface.onnx
│    ├─ DlibResnetExtractor()        dlib.face_recognition_model_v1
│    ├─ FaceNetExtractor()           InceptionResnetV1, load_state_dict (logits stripped)
│    └─ SigLIP2Extractor()          AutoModel + AutoImageProcessor from HF cache
│
├─ iter_dataset(DATA_ROOT)           yield (identity, image_path) for every image
│
└─ for each (identity, image_path):
     │
     ├─ load_image(path)             cv2.imread → BGR uint8
     ├─ detect_faces(img)            InsightFace RetinaFace → [FaceDetection]
     ├─ largest_or_best_face()       pick highest-confidence detection
     ├─ db.insert_face(...)          → faces table (bbox, landmarks, det_score)
     │
     └─ for each model in [arcface, dlib_resnet, facenet, siglip2]:
          │
          ├─ align_face(img, face, model)
          │    ├─ arcface     → _similarity_transform_align (InsightFace landmarks → 112×112 template)
          │    ├─ dlib_resnet → _dlib_chip_align (dlib shape_predictor → get_face_chip → 150×150)
          │    ├─ facenet     → _bbox_margin_crop (margin=0.25 → 160×160)
          │    └─ siglip2    → _bbox_margin_crop (margin=0.40 → 224×224)
          │
          ├─ extractor.extract(aligned_crop)   model-specific preprocess + forward pass
          ├─ l2_normalize(raw_vec)             identical normalization across all models
          └─ db.insert_embedding(...)          → embeddings table (face_id, model, vector)

then:

└─ cluster_all_models()
     └─ for each model:
          ├─ db.load_embeddings_df(model)      load all vectors for this model from MySQL
          ├─ run_hdbscan(X)                    HDBSCAN with shared hyperparameters
          ├─ compute_metrics(true, pred)       ARI, NMI, homogeneity, completeness, V
          ├─ db.insert_cluster_labels(...)     → cluster_results table
          └─ export_cluster_folders(...)       → outputs/clusters/<run_label>/<model>/cluster_N/

└─ metrics_summary_table()                    print comparison table
```

---

## Detection and Alignment — Per Model

### ArcFace (112×112, similarity transform)

InsightFace's RetinaFace detector provides 5 landmark points in the order: left eye, right eye, nose tip, left mouth corner, right mouth corner. A similarity transform (4 degrees of freedom: rotation, uniform scale, x-translation, y-translation — no shear, no independent x/y scaling) is estimated via `cv2.estimateAffinePartial2D` using LMEDS (robust to occasional landmark detection noise). The image is warped so these 5 points align to the canonical 112×112 template:

```
left_eye:   (38.29, 51.70)
right_eye:  (73.53, 51.50)
nose:       (56.03, 71.74)
mouth_l:    (41.55, 92.37)
mouth_r:    (70.73, 92.20)
```

This is the standard ArcFace alignment template and is what the model was trained against.

### dlib-ResNet (150×150, dlib chip)

InsightFace's bbox is converted to a `dlib.rectangle`. dlib's 68-point shape predictor (`shape_predictor_68_face_landmarks.dat`) runs within that region to produce a `dlib.full_object_detection` shape object. `dlib.get_face_chip(img_rgb, shape, size=150)` then performs its internal similarity-transform alignment using whichever of the 68 points it considers canonical reference landmarks for face alignment. The `get_face_chip` function is point-count agnostic — it works identically with 5-point or 68-point predictors.

### FaceNet VGGFace2 (160×160, bbox crop)

A square crop is computed centered on the InsightFace bbox, expanded by 25% (margin=0.25) on each side, clipped to image bounds, and resized to 160×160 via bilinear interpolation. No rotation correction or landmark warping is applied. FaceNet (facenet-pytorch, VGGFace2 weights) was trained on loosely-cropped face images and does not require or benefit from tight landmark-based alignment.

### SigLIP2 (224×224, bbox crop, wider margin)

Same strategy as FaceNet but with a wider margin (40%) and 224×224 target size. SigLIP2 is a general vision-language model (CLIP-style, Google's SigLIP architecture) trained on natural image-text pairs — not on face-specific datasets. A generous margin preserves more surrounding context that the model may rely on. The `AutoImageProcessor` handles all normalization (mean/std for ImageNet-like preprocessing) internally.

---

## Database Schema

```sql
CREATE TABLE faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    identity VARCHAR(255) NOT NULL,
    image_path VARCHAR(1024) NOT NULL,
    bbox JSON NOT NULL,
    landmarks JSON NOT NULL,
    det_score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_image_path (image_path(768))
);

CREATE TABLE embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_id INT NOT NULL,
    model VARCHAR(64) NOT NULL,
    dim INT NOT NULL,
    vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
    UNIQUE KEY uq_face_model (face_id, model)
);

CREATE TABLE cluster_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(64) NOT NULL,
    run_label VARCHAR(255) NOT NULL,
    face_id INT NOT NULL,
    cluster_label INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
    UNIQUE KEY uq_run_face (model, run_label, face_id)
);
```

**Useful inspection queries:**

```sql
-- Row counts across all tables
SELECT 'faces' AS tbl, COUNT(*) FROM faces
UNION ALL SELECT 'embeddings', COUNT(*) FROM embeddings
UNION ALL SELECT 'cluster_results', COUNT(*) FROM cluster_results;

-- Embeddings per model (should be equal if extraction completed cleanly)
SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model;

-- Faces missing one or more model's embedding (should return 0 rows)
SELECT face_id, COUNT(DISTINCT model) AS n_models
FROM embeddings GROUP BY face_id HAVING n_models < 4;

-- Cluster purity per model (distinct identities per predicted cluster)
SELECT cr.model, cr.cluster_label,
       COUNT(DISTINCT f.identity) AS distinct_identities,
       COUNT(*) AS n_faces
FROM cluster_results cr
JOIN faces f ON f.id = cr.face_id
WHERE cr.run_label = 'default'
GROUP BY cr.model, cr.cluster_label
ORDER BY cr.model, distinct_identities DESC;
```

---

## Clustering Methodology

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) was selected over k-means and DBSCAN for three reasons specific to this pipeline:

1. **No k required**: the number of identities in a deployment dataset is typically unknown. HDBSCAN discovers cluster count from density structure.
2. **Noise concept**: points that don't fit confidently into any cluster receive label `-1`. This noise fraction is itself a meaningful embedding-quality signal — a model with 45% noise (SigLIP2) is producing embeddings that are less separable than one with 6% noise (ArcFace), and this difference would be hidden by any method that forces all points into clusters.
3. **Identical hyperparameters across all models**: since this is an embedding-quality comparison and not a clustering-method comparison, the same `min_cluster_size=5`, `min_samples=3`, `metric='euclidean'`, `cluster_selection_method='eom'` are applied to all four models. The only thing that varies between columns of the metrics table is the embedding model itself.

L2 normalization before HDBSCAN is standard practice for face embeddings: it maps all vectors onto the unit hypersphere, making Euclidean distance equivalent (in rank order) to cosine distance, while avoiding the need for HDBSCAN's precomputed-matrix mode that cosine distance would otherwise require.

---

## Results and Metrics

Results on the full dataset (8 identities):

| Model | ARI | NMI | Clusters Found | True Identities | Noise % |
|---|---|---|---|---|---|
| ArcFace | **0.780** | **0.860** | 8 | 8 | 14.6% |
| FaceNet (VGGFace2) | 0.573 | 0.732 | 7 | 8 | 19.8% |
| dlib-ResNet | 0.452 | 0.669 | 8 | 8 | 25.0% |
| SigLIP2 | 0.261 | 0.591 | 7 | 8 | 44.8% |

**Interpretation:**

- **ArcFace** leads across all metrics, consistent with its purpose-built metric learning training (ArcFace loss explicitly optimizes angular separation between identity classes in embedding space).
- **FaceNet** performs second, finding 7/8 identities — a minor missed cluster rather than a systematic failure.
- **dlib-ResNet** finds all 8 clusters but with a 25% noise fraction — its 128-dim embedding has less representational capacity than the 512-dim models, which becomes visible at scale (8 identities × larger dataset vs. the 4-identity, 50-image test subset).
- **SigLIP2** at 0.261 ARI with 44.8% noise reflects what is expected: it is a general-purpose vision-language model not trained for face identity discrimination. These results quantify "how well does a state-of-the-art general vision embedding do at face clustering" — which is itself a useful baseline for the comparison.

**50-image test subset vs. full dataset:** all models scored substantially higher on the 50-image test (ArcFace 0.914, FaceNet 0.846, dlib 0.779, SigLIP2 0.632) than the full dataset. This is expected — a small, randomly-sampled subset is more likely to be clean and well-separated. The full-dataset numbers are the ones to report.

---

## Setup and Running

### Requirements

```bash
pip install -r requirements.txt
# Note: use `onnxruntime` not `onnxruntime-gpu` on macOS
```

### Configuration

Edit `config.py`:
- Set `DATA_ROOT` to your dataset root (`data/raw/` by default)
- Set all 4 weight file paths to their actual locations on your machine
- Set `MYSQL_CONFIG` credentials (or use environment variables `FACE_DB_HOST`, `FACE_DB_USER`, `FACE_DB_PASSWORD`, `FACE_DB_NAME`)
- Set `DETECTOR_CTX_ID = -1` on macOS (no NVIDIA CUDA); `0` on Linux/Windows with a GPU

### Running

```bash
# Quick test on 50 images (recommended before full run)
python run_pipeline.py --limit 50

# Full dataset
python run_pipeline.py

# Re-cluster only (without re-extracting embeddings)
python run_pipeline.py --skip-extraction

# View results
streamlit run app.py
```

### Platform Notes

- **macOS**: `DETECTOR_CTX_ID = -1`; use `onnxruntime` (not `onnxruntime-gpu`). ArcFace runs on CoreML execution provider for acceleration. FaceNet/SigLIP2 run on CPU.
- **Linux/Windows with NVIDIA GPU**: `DETECTOR_CTX_ID = 0`; use `onnxruntime-gpu`. FaceNet/SigLIP2 automatically use CUDA if `torch.cuda.is_available()`.

---

## Known Issues and Fixes Applied

| Issue | Root Cause | Fix Applied |
|---|---|---|
| ArcFace ARI dropped 0.777 → 0.28 | Raw/unaligned crops fed to ArcFace, skipping InsightFace detection and 5-point landmark alignment | Added per-model alignment branches in `common.py`; ArcFace now gets proper norm_crop |
| `RuntimeError: Unexpected key(s) logits.weight, logits.bias` | `facenet_vggface2.pt` checkpoint includes classifier head; model instantiated with `classify=False` has no `logits` layer | Strip `logits.*` keys from state_dict before `load_state_dict()` |
| `AttributeError: 'BaseModelOutputWithPooling' has no attribute 'cpu'` | Newer `transformers` versions return a wrapper object from `get_image_features()` instead of a raw tensor | Check for `.pooler_output` attribute and unwrap when present |
| `UserWarning: CUDAExecutionProvider not in available providers` | ArcFace ONNX session initialized with CUDA provider on macOS | Provider list now checks `ort.get_available_providers()` first; uses CoreML on Mac |
| `shape_predictor_5_face_landmarks.dat` not available | Only the 68-point predictor was present | Updated `DLIB_SHAPE_PREDICTOR_PATH` to the 68-point file; `dlib.get_face_chip()` is point-count agnostic |
