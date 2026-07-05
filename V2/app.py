import os
import cv2
import json
import base64
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import mysql.connector
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
import hdbscan
import plotly.graph_objects as go
import plotly.express as px

# ================================================================
# ⚙️  CONFIGURATION
# ================================================================
DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_v2"
}

BASE         = "/Users/anshumaansinghrathore/Desktop/Face Clustering"
DATASET_DIR  = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"
ARCFACE_PATH = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/Arcface.onnx"
YOLO_PATH    = "/Users/anshumaansinghrathore/Desktop/Face Clustering/models/yolov8n-face.pt"

ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

T1_AUTO_MERGE = 0.30
T2_PROMPT     = 0.55

# ================================================================
# 🎨  PAGE CONFIG + CSS
# ================================================================
st.set_page_config(
    page_title   = "Face Clustering",
    page_icon    = "🧑",
    layout       = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
/* Dark theme overrides */
.stApp { background-color: #0f0f17; }

/* Metric banner */
.metric-banner {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 120px;
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -1px;
}
.metric-label {
    font-size: 0.75rem;
    color: #888;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.good  { color: #4ade80; }
.mid   { color: #facc15; }
.bad   { color: #f87171; }

/* Cluster folder header */
.folder-header {
    background: #1a1a2e;
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 20px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.folder-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.folder-badge {
    background: #6366f1;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.noise-header {
    border-left-color: #f87171;
}
.noise-badge {
    background: #f87171;
}

/* Image card */
.img-card {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
    text-align: center;
}
.img-label {
    font-size: 0.65rem;
    color: #888;
    padding: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Recommendation card */
.rec-card {
    background: #1a1a2e;
    border: 2px solid #6366f1;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}
.rec-card.auto   { border-color: #4ade80; }
.rec-card.prompt { border-color: #facc15; }
.rec-card.new    { border-color: #f87171; }

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 8px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d2d4e;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# 🔌  DATABASE HELPERS
# ================================================================
def get_conn():
    return mysql.connector.connect(**DB_CONFIG)

@st.cache_data(ttl=30, show_spinner=False)
def load_all_embeddings():
    conn   = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, file_name, identity_label, embedding_json,
               is_profile, eye_distance, detector_source, det_confidence
        FROM   face_embeddings_v2
        ORDER  BY identity_label, file_name
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return rows

@st.cache_data(ttl=30, show_spinner=False)
def load_centroids():
    conn   = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT cluster_id, identity_label, centroid_json, member_count
        FROM   cluster_centroids ORDER BY cluster_id
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return rows

def save_centroid_update(cluster_id, new_centroid, new_count, identity):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE cluster_centroids
        SET centroid_json=%s, member_count=%s, identity_label=%s
        WHERE cluster_id=%s
    """, (json.dumps(new_centroid.tolist()), new_count, identity, cluster_id))
    conn.commit(); cursor.close(); conn.close()

def insert_centroid(identity, centroid, count):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(MAX(cluster_id),-1)+1 FROM cluster_centroids"
    )
    new_id = cursor.fetchone()[0]
    cursor.execute("""
        INSERT INTO cluster_centroids
            (cluster_id, identity_label, centroid_json, member_count)
        VALUES (%s,%s,%s,%s)
    """, (new_id, identity, json.dumps(centroid.tolist()), count))
    conn.commit(); cursor.close(); conn.close()
    return new_id

def log_merge(new_cid, matched_cid, dist, decision):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO merge_decisions
            (new_cluster_id, matched_cluster_id, cosine_distance, decision)
        VALUES (%s,%s,%s,%s)
    """, (new_cid, matched_cid, dist, decision))
    conn.commit(); cursor.close(); conn.close()

def insert_new_embedding(file_name, identity, embedding,
                         is_profile, eye_dist, source, conf):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO face_embeddings_v2
            (file_name, identity_label, embedding_json,
             is_profile, eye_distance, detector_source, det_confidence)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (file_name, identity,
          json.dumps(embedding.tolist()),
          bool(is_profile), float(eye_dist), source, float(conf)))
    conn.commit(); cursor.close(); conn.close()


# ================================================================
# 🤖  MODEL LOADING — once per session
# ================================================================
@st.cache_resource(show_spinner="Loading AI models ...")
def load_models():
    yolo   = YOLO(YOLO_PATH)
    sess   = ort.InferenceSession(
        ARCFACE_PATH, providers=["CPUExecutionProvider"]
    )
    retina = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"]
    )
    retina.prepare(ctx_id=-1, det_size=(640,640))
    return yolo, sess, retina

# ================================================================
# 🔧  FACE PROCESSING HELPERS
# ================================================================
def compute_iou(b1, b2):
    ix1 = max(b1[0],b2[0]); iy1 = max(b1[1],b2[1])
    ix2 = min(b1[2],b2[2]); iy2 = min(b1[3],b2[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0.0

def detect_faces(frame, yolo, retina):
    faces = []
    for result in yolo(frame, verbose=False):
        if result.keypoints is None: continue
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i].cpu().numpy())
            if conf < 0.35: continue
            kpts = result.keypoints.data[i].cpu().numpy()
            if len(kpts) < 5: continue
            bbox = result.boxes.xyxy[i].cpu().numpy()
            eye_dist = float(abs(kpts[1][0]-kpts[0][0]))
            faces.append({"bbox":bbox,"kpts":kpts,"conf":conf,
                          "eye_dist":eye_dist,"source":"yolo"})
    for face in retina.get(frame):
        conf = float(face.det_score)
        if conf < 0.35: continue
        kpts = np.hstack([face.kps, np.ones((5,1))])
        eye_dist = float(abs(face.kps[1][0]-face.kps[0][0]))
        faces.append({"bbox":face.bbox,"kpts":kpts,"conf":conf,
                      "eye_dist":eye_dist,"source":"retina"})
    faces.sort(key=lambda x: x["conf"], reverse=True)
    kept=[]
    for f in faces:
        if not any(compute_iou(f["bbox"],k["bbox"])>0.5 for k in kept):
            kept.append(f)
    return kept

def align_face(frame, kpts):
    src = kpts[:5,:2].astype(np.float32)
    M,_ = cv2.estimateAffinePartial2D(src, ARCFACE_SRC, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(frame, M, (112,112),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def embed_face(aligned, sess):
    inp_name = sess.get_inputs()[0].name
    chw  = np.transpose(aligned,(2,0,1)).astype(np.float32)
    norm = (chw - 127.5) / 128.0
    blob = np.ascontiguousarray(np.expand_dims(norm,0))
    raw  = sess.run(None,{inp_name:blob})[0][0]
    n    = np.linalg.norm(raw)
    return (raw/n).astype(np.float32) if n>0 else raw

def spherical_centroid(vecs):
    c = np.mean(vecs, axis=0)
    n = np.linalg.norm(c)
    return c/n if n>0 else c

def cosine_dist(a, b):
    return float(1.0 - np.dot(a,b))


# ================================================================
# 📊  CLUSTERING
# ================================================================
@st.cache_data(ttl=30, show_spinner=False)
def run_hdbscan(embeddings_json_list, min_cls=5, min_smp=1):
    X = np.array([json.loads(e) for e in embeddings_json_list],
                 dtype=np.float32)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cls, min_samples=min_smp,
        metric="euclidean", cluster_selection_method="eom"
    )
    labels = cl.fit_predict(X)
    return X, labels


# ================================================================
# 🖼️  IMAGE DISPLAY HELPERS
# ================================================================
def img_path_for_row(row):
    """Try to find the actual image file on disk for a DB row."""
    identity = row["identity_label"]
    fname    = row["file_name"]
    path     = os.path.join(DATASET_DIR, identity, fname)
    if os.path.exists(path):
        return path
    # Try case-insensitive search
    id_dir = os.path.join(DATASET_DIR, identity)
    if os.path.isdir(id_dir):
        for f in os.listdir(id_dir):
            if f.lower() == fname.lower():
                return os.path.join(id_dir, f)
    return None

def load_pil(path, size=(100,100)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None

def ndarray_to_b64(arr_bgr):
    """Convert cv2 BGR array to base64 PNG string for st.markdown."""
    _, buf = cv2.imencode(".jpg", arr_bgr)
    return base64.b64encode(buf).decode()

def render_cluster_folder(cluster_id, rows_in_cluster,
                          is_noise=False, cols_per_row=8):
    """Render a cluster as a folder with thumbnails."""
    n     = len(rows_in_cluster)
    label = rows_in_cluster[0]["identity_label"] if rows_in_cluster else "?"

    header_cls = "folder-header noise-header" if is_noise else "folder-header"
    badge_cls  = "folder-badge noise-badge"   if is_noise else "folder-badge"
    icon       = "🔕" if is_noise else "📁"
    title      = "Noise (unassigned)" if is_noise else \
                 f"Cluster {cluster_id} — {label}"

    st.markdown(f"""
    <div class="{header_cls}">
        <span style="font-size:1.4rem">{icon}</span>
        <span class="folder-title">{title}</span>
        <span class="{badge_cls}">{n} faces</span>
    </div>
    """, unsafe_allow_html=True)

    # Show up to 32 thumbnails in a grid
    show_rows = rows_in_cluster[:32]
    cols      = st.columns(min(cols_per_row, len(show_rows)))

    for i, row in enumerate(show_rows):
        col = cols[i % cols_per_row]
        path = img_path_for_row(row)
        with col:
            if path:
                img = load_pil(path, (100,100))
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.markdown("🖼️")
            else:
                st.markdown("❓")
            profile_tag = "📐" if row["is_profile"] else ""
            st.caption(f"{profile_tag}{row['file_name'][:12]}")

    if len(rows_in_cluster) > 32:
        st.caption(f"... and {len(rows_in_cluster)-32} more")


# ================================================================
# 📈  METRICS BANNER
# ================================================================
def render_metrics_banner(nmi, ari, sil, noise_pct,
                          n_clusters, n_identities, total):
    def color(val, good, mid):
        return "good" if val>=good else "mid" if val>=mid else "bad"

    metrics = [
        ("NMI",              f"{nmi:.4f}",         color(nmi, 0.7, 0.5)),
        ("ARI",              f"{ari:.4f}",         color(ari, 0.6, 0.4)),
        ("Silhouette",       f"{sil:.4f}",         color(sil, 0.35,0.15)),
        ("Noise",            f"{noise_pct:.1f}%",  color(1-noise_pct/100,0.85,0.7)),
        ("Clusters Found",   str(n_clusters),      "mid"),
        ("Expected",         str(n_identities),    "mid"),
        ("Total Embeddings", str(total),           "mid"),
    ]
    html = '<div class="metric-banner">'
    for label, val, cls in metrics:
        html += f"""
        <div class="metric-card">
            <div class="metric-value {cls}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ================================================================
# 🚀  MAIN APP
# ================================================================
def main():
    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.markdown("## 🧑 Face Clustering")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "📊 Dashboard",
        "🖼️ Cluster Gallery",
        "📤 Upload & Recommend",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**HDBSCAN Parameters**")
    min_cls = st.sidebar.slider("min_cluster_size", 2, 15, 5)
    min_smp = st.sidebar.slider("min_samples",      1,  5, 1)

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # ── Load data ─────────────────────────────────────────────────
    with st.spinner("Loading embeddings ..."):
        rows = load_all_embeddings()

    if not rows:
        st.error("No embeddings found. Run pipeline_v2.py first.")
        return

    df    = pd.DataFrame(rows)
    df["is_profile"] = df["is_profile"].astype(bool)

    emb_json_list = df["embedding_json"].tolist()

    with st.spinner("Clustering ..."):
        X, labels = run_hdbscan(emb_json_list, min_cls, min_smp)

    df["cluster"] = labels

    le            = LabelEncoder()
    ground_truths = le.fit_transform(df["identity_label"].tolist())
    n_identities  = len(le.classes_)
    total         = len(df)
    n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise       = int(np.sum(labels == -1))
    noise_pct     = n_noise / total * 100

    nmi = normalized_mutual_info_score(ground_truths, labels)
    ari = adjusted_rand_score(ground_truths, labels)

    non_noise = labels != -1
    sil = silhouette_score(
        X[non_noise], labels[non_noise], metric="cosine"
    ) if np.sum(non_noise) > 1 else 0.0

    # ================================================================
    # PAGE 1 — DASHBOARD
    # ================================================================
    if page == "📊 Dashboard":
        st.markdown("# 📊 Face Clustering Dashboard")
        st.markdown("---")

        # Metrics banner
        render_metrics_banner(
            nmi, ari, sil, noise_pct,
            n_clusters, n_identities, total
        )

        # ── NMI / ARI gauge charts ────────────────────────────────
        col1, col2, col3 = st.columns(3)

        for col, title, val in [
            (col1, "NMI",        nmi),
            (col2, "ARI",        ari),
            (col3, "Silhouette", sil),
        ]:
            with col:
                color = ("#4ade80" if val>=0.6
                         else "#facc15" if val>=0.35
                         else "#f87171")
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = round(val,4),
                    title = {"text": title, "font":{"color":"#e2e8f0"}},
                    gauge = {
                        "axis"     : {"range":[0,1],
                                      "tickcolor":"#555"},
                        "bar"      : {"color": color},
                        "bgcolor"  : "#1a1a2e",
                        "steps"    : [
                            {"range":[0,   0.35], "color":"#2d0a0a"},
                            {"range":[0.35,0.60], "color":"#2d2510"},
                            {"range":[0.60,1.0],  "color":"#0a2d1a"},
                        ],
                        "threshold": {
                            "line" : {"color":"white","width":2},
                            "value": val
                        }
                    },
                    number={"font":{"color":"#e2e8f0"}}
                ))
                fig.update_layout(
                    height=250,
                    paper_bgcolor="#0f0f17",
                    font_color="#e2e8f0",
                    margin=dict(t=60,b=10,l=20,r=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Frontal vs Profile breakdown ──────────────────────────
        st.markdown('<div class="section-title">Frontal vs Profile Analysis</div>',
                    unsafe_allow_html=True)

        frontal_mask = ~df["is_profile"].values
        profile_mask =  df["is_profile"].values

        sub_results = {}
        for name, mask in [("Frontal", frontal_mask),
                            ("Profile", profile_mask),
                            ("Overall", np.ones(total,bool))]:
            if mask.sum() < 2: continue
            gt  = ground_truths[mask]
            lbl = labels[mask]
            sub_results[name] = {
                "NMI"  : round(normalized_mutual_info_score(gt, lbl),4),
                "ARI"  : round(adjusted_rand_score(gt, lbl),4),
                "Count": int(mask.sum()),
                "Noise%":round(float(np.sum(lbl==-1)/mask.sum()*100),1)
            }

        comp_df = pd.DataFrame([
            {"Category":k,"Metric":"NMI","Value":v["NMI"]}
            for k,v in sub_results.items()
        ]+[
            {"Category":k,"Metric":"ARI","Value":v["ARI"]}
            for k,v in sub_results.items()
        ])

        col_a, col_b = st.columns([2,1])
        with col_a:
            fig2 = px.bar(
                comp_df, x="Category", y="Value",
                color="Metric", barmode="group",
                template="plotly_dark",
                color_discrete_map={"NMI":"#818cf8","ARI":"#4ade80"},
                title="NMI and ARI — Frontal vs Profile"
            )
            fig2.update_layout(
                height=320, yaxis_range=[0,1],
                paper_bgcolor="#0f0f17",
                plot_bgcolor="#0f0f17"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            for name, res in sub_results.items():
                with st.container():
                    st.markdown(f"**{name}** ({res['Count']} faces)")
                    c1,c2 = st.columns(2)
                    c1.metric("NMI", res["NMI"])
                    c2.metric("ARI", res["ARI"])
                    st.caption(f"Noise: {res['Noise%']}%")
                    st.markdown("---")

        # ── Per-identity table ────────────────────────────────────
        st.markdown('<div class="section-title">Per-Identity Breakdown</div>',
                    unsafe_allow_html=True)

        id_rows = []
        for identity in le.classes_:
            mask     = df["identity_label"] == identity
            assigned = sorted(df.loc[mask,"cluster"].unique())
            count    = int(mask.sum())
            noise_n  = int((df.loc[mask,"cluster"]==-1).sum())
            clean    = [c for c in assigned if c!=-1]
            frontal  = int((~df.loc[mask,"is_profile"]).sum())

            status = (
                "✅ Perfect"    if len(clean)==1 and noise_n==0 else
                "~ Partial"    if len(clean)==1 else
                "⚠️ Split"     if len(clean)>1  else
                "🔕 All noise"
            )
            id_rows.append({
                "Identity"       : identity,
                "Total"          : count,
                "Frontal"        : frontal,
                "Profile"        : count-frontal,
                "Clusters"       : len(clean),
                "Noise pts"      : noise_n,
                "Status"         : status
            })

        id_df = pd.DataFrame(id_rows)
        st.dataframe(id_df, use_container_width=True, height=380)

        # ── Embeddings per identity bar ───────────────────────────
        fig3 = px.bar(
            id_df, x="Identity", y="Total",
            color="Status",
            template="plotly_dark",
            title="Embeddings per Identity",
            color_discrete_map={
                "✅ Perfect"   : "#4ade80",
                "~ Partial"   : "#facc15",
                "⚠️ Split"    : "#f87171",
                "🔕 All noise": "#6b7280"
            }
        )
        fig3.update_layout(
            height=320,
            paper_bgcolor="#0f0f17",
            plot_bgcolor="#0f0f17"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ================================================================
    # PAGE 2 — CLUSTER GALLERY
    # ================================================================
    elif page == "🖼️ Cluster Gallery":
        st.markdown("# 🖼️ Cluster Gallery")
        st.markdown("---")

        # Mini metrics banner
        render_metrics_banner(
            nmi, ari, sil, noise_pct,
            n_clusters, n_identities, total
        )

        cols_per_row = st.slider("Thumbnails per row", 4, 12, 8)
        show_profile = st.checkbox("Show profile face indicator", value=True)

        st.markdown("---")

        # ── Sort clusters by size ─────────────────────────────────
        unique_clusters = sorted(
            [c for c in set(labels) if c != -1],
            key=lambda c: -np.sum(labels == c)
        )

        # ── Render each cluster ───────────────────────────────────
        for cluster_id in unique_clusters:
            mask         = df["cluster"] == cluster_id
            cluster_rows = df[mask].to_dict("records")

            # Dominant identity in this cluster
            identities   = [r["identity_label"] for r in cluster_rows]
            dominant     = max(set(identities), key=identities.count)
            purity       = identities.count(dominant) / len(identities) * 100

            # Folder header
            n = len(cluster_rows)
            st.markdown(f"""
            <div class="folder-header">
                <span style="font-size:1.4rem">📁</span>
                <span class="folder-title">
                    Cluster {cluster_id} — {dominant}
                </span>
                <span class="folder-badge">{n} faces</span>
                <span style="color:#888;font-size:0.8rem;margin-left:8px">
                    purity {purity:.0f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Thumbnails
            show_rows = cluster_rows[:cols_per_row * 4]
            n_cols    = min(cols_per_row, len(show_rows))
            cols      = st.columns(n_cols)

            for i, row in enumerate(show_rows):
                col  = cols[i % n_cols]
                path = img_path_for_row(row)
                with col:
                    if path:
                        img = load_pil(path, (110, 110))
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown("🖼️")
                    else:
                        st.markdown("❓")

                    # Caption
                    tags = []
                    if show_profile and row["is_profile"]:
                        tags.append("📐")
                    tags.append(row["file_name"][:14])
                    if row["identity_label"] != dominant:
                        tags.append(f"⚠️{row['identity_label'][:8]}")
                    st.caption(" ".join(tags))

            if len(cluster_rows) > cols_per_row * 4:
                st.caption(
                    f"Showing {cols_per_row*4} of "
                    f"{len(cluster_rows)} — "
                    f"{len(cluster_rows)-cols_per_row*4} more not shown"
                )
            st.markdown("---")

        # ── Noise section ─────────────────────────────────────────
        noise_mask = df["cluster"] == -1
        if noise_mask.any():
            noise_rows = df[noise_mask].to_dict("records")
            n_noise_imgs = len(noise_rows)

            st.markdown(f"""
            <div class="folder-header noise-header">
                <span style="font-size:1.4rem">🔕</span>
                <span class="folder-title">
                    Noise — Unassigned Faces
                </span>
                <span class="folder-badge noise-badge">
                    {n_noise_imgs} faces
                </span>
                <span style="color:#888;font-size:0.8rem;margin-left:8px">
                    {noise_pct:.1f}% of total embeddings
                </span>
            </div>
            """, unsafe_allow_html=True)

            show_rows = noise_rows[:cols_per_row * 3]
            n_cols    = min(cols_per_row, len(show_rows))
            cols      = st.columns(n_cols)

            for i, row in enumerate(show_rows):
                col  = cols[i % n_cols]
                path = img_path_for_row(row)
                with col:
                    if path:
                        img = load_pil(path, (110,110))
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown("🖼️")
                    else:
                        st.markdown("❓")
                    tags = []
                    if show_profile and row["is_profile"]:
                        tags.append("📐")
                    tags.append(row["identity_label"][:12])
                    st.caption(" ".join(tags))

            if len(noise_rows) > cols_per_row * 3:
                st.caption(
                    f"Showing {cols_per_row*3} of {n_noise_imgs}"
                )

    # ================================================================
    # PAGE 3 — UPLOAD & RECOMMEND
    # ================================================================
    elif page == "📤 Upload & Recommend":
        st.markdown("# 📤 Upload Image — Get Cluster Recommendation")
        st.markdown("---")

        # Mini metrics
        render_metrics_banner(
            nmi, ari, sil, noise_pct,
            n_clusters, n_identities, total
        )

        st.markdown("---")
        st.markdown("### Upload a face image")
        st.markdown(
            "The system will detect the face, generate an ArcFace embedding, "
            "compare it to all existing cluster centroids, and recommend "
            "whether to merge into an existing cluster or create a new one."
        )

        uploaded = st.file_uploader(
            "Choose an image", type=["jpg","jpeg","png"],
            help="Upload a photo containing a face"
        )

        t1_val = st.slider(
            "T1 — Auto-merge threshold (d < T1 = definitely same person)",
            0.10, 0.45, T1_AUTO_MERGE, 0.01
        )
        t2_val = st.slider(
            "T2 — Prompt threshold (T1 ≤ d < T2 = ambiguous)",
            t1_val+0.01, 0.80, T2_PROMPT, 0.01
        )

        if uploaded:
            # ── Read uploaded image ───────────────────────────────
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col_img, col_info = st.columns([1,2])
            with col_img:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Uploaded image",
                    use_container_width=True
                )

            with col_info:
                st.markdown("**Detecting faces ...**")
                yolo_m, arcface_sess, retina_m = load_models()
                faces = detect_faces(frame, yolo_m, retina_m)

                if not faces:
                    st.error(
                        "❌ No face detected in the uploaded image. "
                        "Please upload a clearer photo."
                    )
                    return

                st.success(f"✅ {len(faces)} face(s) detected")

                # Use the largest face
                frame_area = frame.shape[0] * frame.shape[1]
                best_face  = max(faces, key=lambda f: (
                    (f["bbox"][2]-f["bbox"][0]) *
                    (f["bbox"][3]-f["bbox"][1]) / frame_area
                ))

                eye_dist   = best_face["eye_dist"]
                is_profile = eye_dist < 20
                source     = best_face["source"]
                conf       = best_face["conf"]

                st.markdown(f"""
                - **Detector:** {source}
                - **Confidence:** {conf:.3f}
                - **Eye distance:** {eye_dist:.1f} px
                - **Face type:** {"📐 Profile" if is_profile else "👁️ Frontal"}
                """)

                # Align + embed
                aligned = align_face(frame, best_face["kpts"])
                if aligned is None:
                    st.error("❌ Face alignment failed.")
                    return

                st.image(
                    cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
                    caption="Aligned 112×112 patch → ArcFace input",
                    width=112
                )

                new_emb = embed_face(aligned, arcface_sess)

            # ── Load centroids and compute distances ──────────────
            centroid_rows_db = load_centroids()
            if not centroid_rows_db:
                st.warning(
                    "No cluster centroids found. "
                    "Run clustering_v2.py first."
                )
                return

            centroids = []
            for row in centroid_rows_db:
                centroids.append({
                    "cluster_id"    : row["cluster_id"],
                    "identity_label": row["identity_label"],
                    "centroid"      : np.array(
                        json.loads(row["centroid_json"]),
                        dtype=np.float32
                    ),
                    "member_count"  : row["member_count"]
                })

            # Compute distances to all centroids
            distances = sorted([
                (cosine_dist(new_emb, c["centroid"]), c)
                for c in centroids
            ], key=lambda x: x[0])

            best_dist, best_cluster = distances[0]

            st.markdown("---")
            st.markdown("### 🎯 Recommendation")

            # Distance bar chart
            dist_df = pd.DataFrame([{
                "Cluster"  : f"C{c['cluster_id']} {c['identity_label'][:15]}",
                "Distance" : round(d, 4)
            } for d, c in distances[:10]])

            fig = px.bar(
                dist_df, x="Cluster", y="Distance",
                template="plotly_dark",
                title="Cosine Distance to Top 10 Clusters",
                color="Distance",
                color_continuous_scale="RdYlGn_r",
            )
            fig.add_hline(
                y=t1_val, line_dash="dash",
                line_color="#4ade80",
                annotation_text=f"T1={t1_val} (auto-merge)"
            )
            fig.add_hline(
                y=t2_val, line_dash="dash",
                line_color="#facc15",
                annotation_text=f"T2={t2_val} (prompt)"
            )
            fig.update_layout(
                height=340,
                paper_bgcolor="#0f0f17",
                plot_bgcolor="#0f0f17",
                yaxis_range=[0,1]
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Decision routing ──────────────────────────────────
            if best_dist < t1_val:
                # T1 — Auto merge
                card_cls = "auto"
                decision_text = "🟢 AUTO-MERGE"
                color_hex     = "#4ade80"
                explanation   = (
                    f"Distance **{best_dist:.4f}** is below T1 ({t1_val}). "
                    f"This face is almost certainly the same person as "
                    f"**{best_cluster['identity_label']}**."
                )
                btn_label = (
                    f"✅ Confirm Auto-Merge into "
                    f"'{best_cluster['identity_label']}'"
                )
                can_merge = True

            elif best_dist < t2_val:
                # T2 — Prompt
                card_cls = "prompt"
                decision_text = "🟡 USER VERIFICATION NEEDED"
                color_hex     = "#facc15"
                explanation   = (
                    f"Distance **{best_dist:.4f}** is in the ambiguous zone "
                    f"({t1_val}–{t2_val}). "
                    f"Closest cluster is **{best_cluster['identity_label']}** "
                    f"— please verify."
                )
                btn_label = (
                    f"✅ Yes — Same as '{best_cluster['identity_label']}'"
                )
                can_merge = True

            else:
                # T3 — New cluster
                card_cls = "new"
                decision_text = "🔴 NEW IDENTITY"
                color_hex     = "#f87171"
                explanation   = (
                    f"Distance **{best_dist:.4f}** exceeds T2 ({t2_val}). "
                    f"This face does not match any existing cluster. "
                    f"A new identity cluster will be created."
                )
                btn_label = None
                can_merge = False

            st.markdown(f"""
            <div class="rec-card {card_cls}">
                <div style="font-size:1.2rem;font-weight:700;
                            color:{color_hex};margin-bottom:8px">
                    {decision_text}
                </div>
                <div style="color:#cbd5e1">{explanation}</div>
                <div style="margin-top:12px;color:#94a3b8;font-size:0.85rem">
                    Closest cluster: <b>{best_cluster['identity_label']}</b>
                    &nbsp;|&nbsp;
                    Members: <b>{best_cluster['member_count']}</b>
                    &nbsp;|&nbsp;
                    Distance: <b>{best_dist:.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Action buttons ────────────────────────────────────
            col_btn1, col_btn2 = st.columns(2)

            if can_merge:
                with col_btn1:
                    if st.button(btn_label, type="primary"):
                        # Weighted centroid update
                        n_old    = best_cluster["member_count"]
                        old_c    = best_cluster["centroid"]
                        merged   = n_old * old_c + 1 * new_emb
                        norm     = np.linalg.norm(merged)
                        merged   = merged / norm if norm > 0 else merged

                        save_centroid_update(
                            best_cluster["cluster_id"],
                            merged,
                            n_old + 1,
                            best_cluster["identity_label"]
                        )
                        insert_new_embedding(
                            uploaded.name,
                            best_cluster["identity_label"],
                            new_emb,
                            is_profile, eye_dist, source, conf
                        )
                        dec = ("auto_merge" if best_dist < t1_val
                               else "user_confirmed")
                        log_merge(
                            None,
                            best_cluster["cluster_id"],
                            best_dist, dec
                        )
                        st.cache_data.clear()
                        st.success(
                            f"✅ Merged into '{best_cluster['identity_label']}'. "
                            f"Cluster now has {n_old+1} members. "
                            f"Metrics updated."
                        )
                        st.rerun()

                with col_btn2:
                    reject_label = "❌ No — Create new cluster instead"
                    if st.button(reject_label):
                        new_name = st.text_input(
                            "Enter identity name for new cluster:"
                        )
                        if new_name:
                            new_id = insert_centroid(
                                new_name, new_emb, 1
                            )
                            insert_new_embedding(
                                uploaded.name, new_name,
                                new_emb, is_profile,
                                eye_dist, source, conf
                            )
                            log_merge(
                                new_id,
                                best_cluster["cluster_id"],
                                best_dist, "user_rejected"
                            )
                            st.cache_data.clear()
                            st.success(
                                f"🆕 New cluster created: '{new_name}'"
                            )
                            st.rerun()
            else:
                # T3 — register new directly
                new_name = st.text_input(
                    "Enter name for this new identity:"
                )
                if st.button("🆕 Register as New Identity",
                             type="primary") and new_name:
                    new_id = insert_centroid(new_name, new_emb, 1)
                    insert_new_embedding(
                        uploaded.name, new_name,
                        new_emb, is_profile,
                        eye_dist, source, conf
                    )
                    log_merge(
                        new_id,
                        best_cluster["cluster_id"],
                        best_dist, "new_cluster"
                    )
                    st.cache_data.clear()
                    st.success(
                        f"🆕 Registered '{new_name}' as new cluster."
                    )
                    st.rerun()

            # ── Updated metrics preview ───────────────────────────
            st.markdown("---")
            st.markdown("### 📈 Current Benchmarking Metrics")
            render_metrics_banner(
                nmi, ari, sil, noise_pct,
                n_clusters, n_identities, total
            )
            st.caption(
                "Metrics update automatically after each merge. "
                "ARI and NMI will improve as split clusters are merged."
            )


if __name__ == "__main__":
    main()
