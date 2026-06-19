# Face Embedding Comparison Pipeline

Compares embedding quality from 4 models — **ArcFace**, **dlib-ResNet**, **FaceNet (VGGFace2)**,
**SigLIP2** — using a shared InsightFace detector and identical HDBSCAN clustering, so the
comparison isolates *embedding quality* rather than detection or clustering differences.

## Why this structure

Your ArcFace ARI dropped from 0.777 → 0.28 because the new pipeline fed raw/unaligned crops
straight from the dataset loader into ArcFace, instead of running detection + 5-point landmark
alignment first. ArcFace (like most metric-learning face models) is only meaningful on tightly
aligned 112×112 crops — feed it anything else and embeddings degrade badly without erroring out.

This pipeline fixes that by **separating concerns into 3 stages**, each in its own module:

1. **`common.py`** — load image, detect (bbox + 5 landmarks, via InsightFace, run ONCE per image),
   then align **per model's own convention** (ArcFace gets its canonical norm_crop, dlib gets its
   own chip alignment, FaceNet/SigLIP2 get a looser bbox-margin crop). This is the part that was
   silently broken before.
2. **`embeddings.py`** — one extractor class per model behind a common `.extract()` interface.
   L2 normalization happens centrally, after extraction, identically for every model.
3. **`clustering.py`** — HDBSCAN with **identical hyperparameters across all 4 models** (don't
   tune per-model — that breaks the fairness of the comparison), plus ARI/NMI/homogeneity metrics
   and cluster-folder export for visual inspection.

`storage.py` persists everything (bbox, landmarks, embeddings, cluster assignments) to MySQL so
you never have to recompute detection/embeddings to re-cluster or debug.

`run_pipeline.py` orchestrates the full thing. `app.py` is the Streamlit viewer.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py`:
- `DATA_ROOT` — should already match `data/raw/person1/...`
- Model weight paths (`ARCFACE_ONNX_PATH`, `DLIB_RESNET_PATH`, `FACENET_VGGFACE2_PATH`, etc.)
- `DLIB_SHAPE_PREDICTOR_PATH` — **you'll need to download this separately** if you don't have it;
  dlib's recognition model expects dlib's own 5-point landmarks, not InsightFace's. Get it from
  dlib's model repo (`shape_predictor_5_face_landmarks.dat`).
- `MYSQL_CONFIG` — or set env vars `FACE_DB_HOST`, `FACE_DB_USER`, `FACE_DB_PASSWORD`, `FACE_DB_NAME`

## Run

```bash
# Quick test on a small subset first — confirms detection/alignment/extraction work end to end
python run_pipeline.py --limit 50

# Full run
python run_pipeline.py

# Re-cluster without re-extracting (e.g. after changing HDBSCAN params)
python run_pipeline.py --skip-extraction

# View results
streamlit run app.py
```

## Before trusting the numbers

1. Run with `--limit 50` first and check `outputs/clusters/default/arcface/` by eye — do the
   clusters look like the same person? If ArcFace's ARI is still bad after proper alignment,
   *then* something else is wrong (and it's worth checking 5–10 aligned crops visually — bad
   landmarks producing rotated/cropped-wrong faces is the next most common culprit).
2. Confirm `DLIB_SHAPE_PREDICTOR_PATH` actually points to a valid file — dlib's chip alignment
   will fail loudly if not, which is correct behavior (better than silently producing garbage).
3. Check `n_no_face` in the extraction log — if it's high, your dataset has images InsightFace's
   detector struggles with (low-res, extreme angles, occlusion), which will hurt every model's
   metrics, not just one.
