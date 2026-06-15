import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Direct Pipeline Dashboard", layout="wide")

st.title("🎭 Face Clustering — Direct BBox Pipeline")
st.caption("No landmark alignment — raw bounding box crop → ArcFace. Compare with the aligned pipeline to see alignment's impact.")

# ── Load JSON ─────────────────────────────────────────────────
JSON_FILE   = "clustering_results_direct.json"
RESULTS_DIR = Path("/Users/anshumaansinghrathore/Desktop/Face Clustering/results_direct")

if not Path(JSON_FILE).exists():
    st.error(f"`{JSON_FILE}` not found. Run: `python direct_pipeline.py --mode all` first.")
    st.stop()

with open(JSON_FILE) as f:
    data = json.load(f)

best = data["best_params"]

# ── Metrics ───────────────────────────────────────────────────
st.header("📊 Clustering Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("NMI",      f"{best['nmi']:.3f}")
col2.metric("ARI",      f"{best['ari']:.3f}")
col3.metric("Purity",   f"{best['purity']:.3f}")
col4.metric("Clusters", best["n_clusters"])
col5.metric("Noise %",  f"{best['noise_pct']:.2f}")

# ── Comparison note ───────────────────────────────────────────
st.info(
    "**Aligned pipeline** (pipeline_new.py) uses InsightFace landmarks + affine warp. "
    "**This pipeline** skips that and feeds a raw bbox crop directly to ArcFace. "
    "Lower NMI/ARI here = alignment contributes that much to accuracy."
)

st.divider()

# ── Identity Analysis ─────────────────────────────────────────
st.header("👥 Identity Analysis")

rows = []
for identity, info in data["per_identity"].items():
    rows.append({
        "Identity":       identity,
        "Images":         info["total"],
        "Noise Images":   info["noise"],
        "Clusters Found": info["n_clusters"]
    })

df = pd.DataFrame(rows)
st.dataframe(df.sort_values("Identity"), use_container_width=True)

# ── Split identities ──────────────────────────────────────────
st.header("⚠️ Split Identities")
found = False
for identity, info in data["per_identity"].items():
    if info["n_clusters"] > 1:
        found = True
        st.warning(f"{identity} split into {info['n_clusters']} clusters")
if not found:
    st.success("No split identities found")

st.divider()

# ── Cluster Explorer ──────────────────────────────────────────
st.header("📁 Cluster Explorer")

if not RESULTS_DIR.exists():
    st.error("results_direct folder not found. Run: `python direct_pipeline.py --mode folders`")
    st.stop()

cluster_folders = sorted([f.name for f in RESULTS_DIR.iterdir() if f.is_dir()])
selected        = st.selectbox("Select Cluster", cluster_folders)
cluster_path    = RESULTS_DIR / selected

images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    images.extend(cluster_path.glob(ext))

st.write(f"**{len(images)} images** in {selected}")

cols = st.columns(5)
for idx, img_path in enumerate(sorted(images)):
    try:
        img = Image.open(img_path)
        cols[idx % 5].image(img, caption=img_path.name, use_container_width=True)
    except Exception:
        pass

st.divider()

# ── Cluster Summary ───────────────────────────────────────────
st.header("📈 Cluster Summary")

stats = []
for folder in cluster_folders:
    count = len(list((RESULTS_DIR / folder).glob("*")))
    stats.append({"Cluster": folder, "Images": count})

st.dataframe(
    pd.DataFrame(stats).sort_values("Images", ascending=False),
    use_container_width=True
)

st.success("Dashboard Loaded ✅")
