import os
import glob
import base64

import numpy as np
import cv2
import streamlit as st
import pandas as pd

import config
from storage import Storage
from clustering import cluster
from common import detect_faces, largest_or_best_face, align_face
from embedding import ArcFaceExtractor, get_normalized_embedding
from recommendation import (
    recommend_for_embedding,
    assign_face_to_cluster,
    create_new_cluster,
    reclaim_noise,
    pairwise_cluster_scan,
    merge_clusters,
    calibrate_thresholds,
)

st.set_page_config(page_title="Face Clustering & Recommendation", layout="wide")
st.title("Face Clustering & Recommendation System")
st.caption("Detector: InsightFace (buffalo_l)  ·  Embedding: ArcFace (512-d)  ·  Clustering: HDBSCAN")

run_label = st.sidebar.text_input("Run label", value="default")

if st.sidebar.button("Re-run clustering (uses existing DB embeddings)"):
    with st.spinner("Running HDBSCAN..."):
        st.session_state["cluster_result"] = cluster(run_label=run_label)
    st.sidebar.warning(
        "Re-clustering just rebuilt cluster_results + centroids from scratch. "
        "Any manual merges / noise reclamation you'd done for this run_label are now gone -- "
        "that's expected, HDBSCAN has no memory of them."
    )

if "cluster_result" not in st.session_state:
    with st.spinner("Loading clustering results..."):
        try:
            st.session_state["cluster_result"] = cluster(run_label=run_label)
        except ValueError as e:
            st.session_state["cluster_result"] = None
            st.warning(f"{e} Run the extraction pipeline first (runpipeline.py), then reload.")

cluster_result = st.session_state["cluster_result"]


@st.cache_resource(show_spinner=False)
def get_extractor():
    return ArcFaceExtractor()


def _insert_uploaded_face(save_path: str, vec: np.ndarray):
    db = Storage()
    face_id = db.insert_face(
        identity="uploaded", image_path=save_path,
        bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
    )
    db.insert_embedding(face_id, vec)
    return db, face_id


def _save_face_crop(img_bgr: np.ndarray, bbox: np.ndarray, base_name: str, idx: int, margin_frac: float = 0.3) -> str:
    """Crops a visible (non-warped) region around one detected face, with a margin, and
    saves it under a filename unique to this face -- so a group photo's N faces each get
    their own row in `faces` without colliding on the image_path uniqueness constraint."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    nx1 = int(max(0, x1 - bw * margin_frac))
    ny1 = int(max(0, y1 - bh * margin_frac))
    nx2 = int(min(w, x2 + bw * margin_frac))
    ny2 = int(min(h, y2 + bh * margin_frac))
    crop = img_bgr[ny1:ny2, nx1:nx2]

    os.makedirs(config.UPLOADS_ROOT, exist_ok=True)
    save_path = os.path.join(config.UPLOADS_ROOT, f"{base_name}__face{idx}.jpg")
    cv2.imwrite(save_path, crop)
    return save_path


from typing import Optional


@st.cache_data(show_spinner=False)
def _square_tile_data_uri(path: str, max_dim: int = 300):
    """Reads an image from disk and returns a base64 data URI, downscaled so the
    longer side is max_dim -- kept small since it's only ever displayed as a
    fixed-size cropped tile, never at full resolution."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


_NOISE_TILE_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='150' height='150'>"
    "<rect width='150' height='150' fill='%23333'/>"
    "<text x='50%25' y='50%25' fill='%23999' font-size='14' "
    "text-anchor='middle' dominant-baseline='middle'>Noise</text></svg>"
)


def _render_square_tile(image_path: Optional[str], size_px: int = 150):
    """Renders a uniform, center-cropped square tile -- same size and shape no
    matter whether the source image is a tall portrait, wide landscape, huge,
    or tiny. This is what keeps the grid looking like a real grid."""
    uri = None
    if image_path and os.path.exists(image_path):
        uri = _square_tile_data_uri(image_path)
    if uri is None:
        uri = _NOISE_TILE_SVG
    st.markdown(
        f"<div style='width:100%;aspect-ratio:1/1;overflow:hidden;border-radius:8px;'>"
        f"<img src='{uri}' style='width:100%;height:100%;object-fit:cover;display:block;'/>"
        f"</div>",
        unsafe_allow_html=True,
    )


tab_metrics, tab_flow1, tab_flow2, tab_noise, tab_calibrate = st.tabs([
    "Metrics & clusters",
    "Recommend (Flow 1)",
    "Pairwise check (Flow 2)",
    "Noise reclamation",
    "Calibrate thresholds",
])

# ===========================================================================
# TAB: Metrics + cluster browser
# ===========================================================================
with tab_metrics:
    if cluster_result is None:
        st.info("No embeddings in the database yet.")
    else:
        st.subheader("Clustering metrics")
        metrics = cluster_result["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ARI", f"{metrics['ari']:.3f}")
        m2.metric("NMI", f"{metrics['nmi']:.3f}")
        m3.metric("Clusters found", metrics["n_predicted_clusters"])
        m4.metric("Noise fraction", f"{metrics['noise_fraction']:.1%}")
        with st.expander("Full metrics"):
            st.json(metrics)

        st.subheader("Cluster browser")
        st.caption("Live view from the database -- reflects Flow 1, Flow 2, and noise-reclamation "
                   "changes immediately, including any new clusters they create.")

        db = Storage()
        cluster_labels = db.get_all_cluster_labels_including_noise(run_label)
        centroids_df = db.load_centroids_df(run_label)
        db.close()

        if not cluster_labels:
            st.warning("No clusters found for this run_label yet. Run the pipeline / clustering first.")
        else:
            ordered_labels = sorted(cluster_labels, key=lambda l: (l == -1, l))
            centroid_lookup = (
                {int(r["cluster_label"]): r for _, r in centroids_df.iterrows()}
                if not centroids_df.empty else {}
            )

            if ("browse_cluster_label" not in st.session_state
                    or st.session_state["browse_cluster_label"] not in ordered_labels):
                st.session_state["browse_cluster_label"] = ordered_labels[0]

            n_cols = 6
            grid_cols = st.columns(n_cols)
            for i, label in enumerate(ordered_labels):
                with grid_cols[i % n_cols]:
                    if label == -1:
                        _render_square_tile(None)
                        tile_caption = "Noise"
                    else:
                        row = centroid_lookup.get(label)
                        thumb_path = row["representative_image_path"] if row is not None else None
                        identity = row["representative_identity"] if row is not None else "?"
                        _render_square_tile(thumb_path)
                        tile_caption = f"Cluster {label} ({identity})"

                    is_selected = st.session_state["browse_cluster_label"] == label
                    if st.button(tile_caption, key=f"grid_select_{label}",
                                 use_container_width=True,
                                 type="primary" if is_selected else "secondary"):
                        st.session_state["browse_cluster_label"] = label
                        st.rerun()

            chosen_label = st.session_state["browse_cluster_label"]
            st.markdown("---")
            st.write(f"### {'Noise' if chosen_label == -1 else f'Cluster {chosen_label}'}")

            db = Storage()
            face_rows = db.get_cluster_faces_info(run_label, chosen_label)
            db.close()

            st.write(f"**{len(face_rows)} images**")

            identities_in_cluster = sorted({r["identity"] for r in face_rows})
            if len(identities_in_cluster) > 1:
                st.error(f"Mixed identities in this cluster: {identities_in_cluster}")
            elif identities_in_cluster:
                st.success(f"Pure cluster: {identities_in_cluster[0]}")

            detail_cols = st.columns(6)
            for i, row in enumerate(face_rows):
                with detail_cols[i % 6]:
                    _render_square_tile(row["image_path"])
                    st.caption(row["identity"])

# ===========================================================================
# TAB: Flow 1 -- upload a new face, compare to all centroids
# ===========================================================================
with tab_flow1:
    st.subheader("Upload one or more images and get cluster recommendations")
    st.caption("Each image is analyzed independently -- some may auto-merge, some may ask you, "
               "some may form new clusters, all in the same batch.")
    uploaded_files = st.file_uploader("Upload images, one face each", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="flow1_upload")

    if uploaded_files and st.button("Analyze", key="flow1_analyze"):
        extractor = get_extractor()

        st.session_state["flow1_result_messages"] = []
        st.session_state["flow1_pending_items"] = {}
        st.session_state["flow1_diagnostics"] = []

        for uf in uploaded_files:
            raw_bytes = uf.getvalue()
            img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

            faces = detect_faces(img_bgr)
            face = largest_or_best_face(faces)

            if face is None:
                st.session_state["flow1_result_messages"].append(
                    ("error", f"{uf.name}: no face detected.")
                )
                continue

            aligned = align_face(img_bgr, face)
            vec = get_normalized_embedding(extractor, aligned)

            # Diagnostic: which face (of possibly several) got used in THIS file, and
            # what did the actual aligned crop look like -- catches a wrong face being
            # picked out of a multi-face photo (e.g. two people in frame).
            st.session_state["flow1_diagnostics"].append({
                "label": uf.name,
                "n_faces": len(faces),
                "det_score": face.det_score,
                "aligned_rgb": cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
            })

            db = Storage()
            rec = recommend_for_embedding(vec, run_label, db)
            db.close()

            os.makedirs(config.UPLOADS_ROOT, exist_ok=True)
            save_path = os.path.join(config.UPLOADS_ROOT, uf.name)
            with open(save_path, "wb") as f:
                f.write(raw_bytes)

            # ---------------------------------------------------------
            # Strict three-way rule, applied per file: only ask_user
            # waits for a human decision; auto_merge/new_cluster act
            # immediately, independently, for every file in the batch.
            # ---------------------------------------------------------
            if rec["status"] == "auto_merge":
                db2, face_id = _insert_uploaded_face(save_path, vec)
                assign_face_to_cluster(face_id, run_label, rec["nearest_cluster_label"], vec, db2)
                db2.close()
                st.session_state["flow1_result_messages"].append((
                    "success",
                    f"{uf.name}: clustered into cluster {rec['nearest_cluster_label']} "
                    f"(identity: {rec['representative_identity']}) -- auto-merged "
                    f"(distance {rec['distance']:.3f})."
                ))
            elif rec["status"] == "new_cluster":
                db2, face_id = _insert_uploaded_face(save_path, vec)
                new_label = create_new_cluster(face_id, run_label, vec, db2)
                db2.close()
                dist_note = f" (nearest existing cluster was {rec['distance']:.3f} away)" if rec["distance"] is not None else ""
                st.session_state["flow1_result_messages"].append((
                    "info",
                    f"{uf.name}: no existing cluster matched{dist_note} -- new cluster {new_label} formed."
                ))
            else:  # ask_user
                st.session_state["flow1_pending_items"][save_path] = {
                    "label": uf.name, "rec": rec, "vec": vec, "save_path": save_path,
                }

    # Diagnostics for the whole batch, one entry per uploaded file that had a detectable face.
    if st.session_state.get("flow1_diagnostics"):
        with st.expander(f"Diagnostics for {len(st.session_state['flow1_diagnostics'])} analyzed image(s)"):
            for diag in st.session_state["flow1_diagnostics"]:
                st.write(f"**{diag['label']}**")
                if diag["n_faces"] > 1:
                    st.warning(f"{diag['n_faces']} faces detected -- used the highest-confidence one "
                               f"(det_score={diag['det_score']:.3f}).")
                st.image(diag["aligned_rgb"], width=150, caption="112x112 crop fed into ArcFace")
                st.markdown("---")

    for kind, msg in st.session_state.get("flow1_result_messages", []):
        getattr(st, kind)(msg)

    pending_items = st.session_state.get("flow1_pending_items", {})
    for item_key, item in list(pending_items.items()):
        rec = item["rec"]
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            _render_square_tile(item["save_path"])
            st.caption(f"{item['label']} (uploaded)")
        with col_b:
            _render_square_tile(rec["representative_image_path"])
            st.caption(f"Nearest cluster {rec['nearest_cluster_label']} "
                       f"(identity: {rec['representative_identity']})")

        if rec.get("auto_merge_demoted"):
            st.warning(f"{item['label']}: asking for confirmation -- distance to cluster "
                       f"{rec['nearest_cluster_label']} is **{rec['distance']:.3f}** (under the global "
                       f"auto-merge ceiling, but farther than this specific cluster has ever spread "
                       f"before). Same person?")
        else:
            st.warning(f"{item['label']}: asking for confirmation -- cosine distance to cluster "
                       f"{rec['nearest_cluster_label']} is **{rec['distance']:.3f}**. Same person?")

        c1, c2 = st.columns(2)
        safe_key = abs(hash(item_key))
        if c1.button("Yes, same person", key=f"flow1_yes_{safe_key}"):
            db, face_id = _insert_uploaded_face(item["save_path"], item["vec"])
            assign_face_to_cluster(face_id, run_label, rec["nearest_cluster_label"], item["vec"], db)
            db.close()
            st.session_state["flow1_result_messages"].append((
                "success",
                f"{item['label']}: clustered into cluster {rec['nearest_cluster_label']} "
                f"(identity: {rec['representative_identity']})."
            ))
            del st.session_state["flow1_pending_items"][item_key]
            st.rerun()
        if c2.button("No, different person", key=f"flow1_no_{safe_key}"):
            db, face_id = _insert_uploaded_face(item["save_path"], item["vec"])
            new_label = create_new_cluster(face_id, run_label, item["vec"], db)
            db.close()
            st.session_state["flow1_result_messages"].append((
                "info",
                f"{item['label']}: no match -- new cluster {new_label} has been formed."
            ))
            del st.session_state["flow1_pending_items"][item_key]
            st.rerun()

# ===========================================================================
# TAB: Flow 2 -- pairwise cluster-vs-cluster sweep
# ===========================================================================
with tab_flow2:
    st.subheader("Scan existing clusters for suspicious pairs (possible same person, split across clusters)")

    if st.button("Scan for suspicious cluster pairs", key="flow2_scan"):
        db = Storage()
        st.session_state["flow2_suspicious"] = pairwise_cluster_scan(run_label, db)
        db.close()

    if "flow2_suspicious" in st.session_state:
        suspicious = st.session_state["flow2_suspicious"]

        if not suspicious:
            st.success("No suspicious cluster pairs found within the threshold zone.")
        else:
            st.write(f"Found **{len(suspicious)}** suspicious pair(s).")
            for pair in suspicious:
                key = f"{pair['cluster_a']}_{pair['cluster_b']}"
                st.markdown("---")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    _render_square_tile(pair["rep_a_image_path"])
                    st.caption(f"Cluster {pair['cluster_a']}")
                with c2:
                    _render_square_tile(pair["rep_b_image_path"])
                    st.caption(f"Cluster {pair['cluster_b']}")
                with c3:
                    st.write(f"Distance: **{pair['distance']:.3f}**"
                             + (" (auto-mergeable)" if pair["auto_mergeable"] else ""))
                    b1, b2 = st.columns(2)
                    if b1.button("Merge (same person)", key=f"flow2_merge_{key}"):
                        db = Storage()
                        merge_result = merge_clusters(run_label, pair["cluster_a"], pair["cluster_b"], db)
                        db.close()
                        st.success(f"Merged cluster {merge_result['merged_away']} into "
                                   f"{merge_result['kept']} ({merge_result['n_members']} members).")
                        st.session_state["flow2_suspicious"] = [
                            s for s in suspicious
                            if not (s["cluster_a"] == pair["cluster_a"] and s["cluster_b"] == pair["cluster_b"])
                        ]
                        st.rerun()
                    if b2.button("Not the same person", key=f"flow2_reject_{key}"):
                        st.session_state["flow2_suspicious"] = [
                            s for s in suspicious
                            if not (s["cluster_a"] == pair["cluster_a"] and s["cluster_b"] == pair["cluster_b"])
                        ]
                        st.rerun()

# ===========================================================================
# TAB: Noise reclamation
# ===========================================================================
with tab_noise:
    st.subheader("Reclaim noise points (HDBSCAN's -1 bucket)")
    st.caption("Noise points are real faces HDBSCAN couldn't confidently place in any dense cluster -- "
               "not garbage. This routes each one through the same Flow-1 logic against real clusters.")

    if st.button("Scan noise points", key="noise_scan"):
        db = Storage()
        st.session_state["noise_result"] = reclaim_noise(run_label, db, auto_apply=True)
        db.close()

    if "noise_result" in st.session_state:
        result = st.session_state["noise_result"]
        st.write(f"**{len(result['auto_merged'])}** auto-merged, "
                 f"**{len(result['suggestions'])}** need your review, "
                 f"**{len(result['left_as_noise'])}** left as noise.")

        remaining_suggestions = []
        for s in result["suggestions"]:
            st.markdown("---")
            db = Storage()
            face_info = db.get_face_info(s["face_id"])
            db.close()

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                _render_square_tile(face_info["image_path"])
                st.caption("Noise point")
            with c2:
                _render_square_tile(s["representative_image_path"])
                st.caption(f"Cluster {s['nearest_cluster_label']}")
            with c3:
                st.write(f"Distance: **{s['distance']:.3f}**")
                b1, b2 = st.columns(2)
                resolved = False
                if b1.button("Yes, same person", key=f"noise_yes_{s['face_id']}"):
                    db = Storage()
                    vec = db.get_embedding_vector(s["face_id"])
                    assign_face_to_cluster(s["face_id"], run_label, s["nearest_cluster_label"], vec, db)
                    db.close()
                    st.success("Assigned to cluster.")
                    resolved = True
                if b2.button("No, leave as noise", key=f"noise_no_{s['face_id']}"):
                    st.info("Left as noise.")
                    resolved = True
                if not resolved:
                    remaining_suggestions.append(s)

        result["suggestions"] = remaining_suggestions
        st.session_state["noise_result"] = result

# ===========================================================================
# TAB: Threshold calibration
# ===========================================================================
with tab_calibrate:
    st.subheader("Data-driven threshold calibration")
    st.caption(
        "config.THRESHOLDS starts with a generic guess. This uses the ground-truth identity "
        "labels already in your dataset (from the data/raw/<identity>/ folders) to suggest better "
        "T1 / T2 values -- an offline tuning step you wouldn't have without labelled data."
    )

    target_rate_pct = st.slider("Target false-merge rate for the precision-based T1", 0.1, 5.0, 1.0, 0.1,
                                 help="e.g. 1.0 means: fewer than 1% of auto-merges at this threshold "
                                      "would actually be joining two different people, based on your data.")

    if st.button("Calibrate", key="calibrate_run"):
        db = Storage()
        try:
            st.session_state["calibration_result"] = calibrate_thresholds(
                run_label, db, target_false_merge_rate=target_rate_pct / 100.0
            )
        except ValueError as e:
            st.warning(str(e))
        db.close()

    if "calibration_result" in st.session_state:
        r = st.session_state["calibration_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("T1 -- percentile method", f"{r['suggested_t1']:.3f}")
        c2.metric("T1 -- precision method (prefer this)", f"{r['suggested_t1_precision']:.3f}")
        c3.metric("Suggested T2 (new cluster above)", f"{r['suggested_t2']:.3f}")
        st.write(f"Same-identity mean distance: {r['same_identity_dist_mean']:.3f}  |  "
                 f"Different-identity mean distance: {r['diff_identity_dist_mean']:.3f}  |  "
                 f"Sampled {r['n_same_pairs']} same-id and {r['n_diff_pairs']} diff-id pairs.")
        st.write(f"At the precision-method T1, the achieved false-merge rate in your sample is "
                 f"**{r['achieved_false_merge_rate']:.2%}**.")
        if r["no_safe_auto_merge_zone"]:
            st.error("Even the closest different-identity pair in your data sits below your target "
                      "false-merge rate -- there's no distance cutoff that's safe to auto-merge at "
                      "without asking. Consider raising the target rate above, or leaving auto-merge "
                      "off (set T1 very low, e.g. 0.0) and relying on the ask_user zone for everything.")
        if not r["clean_separation"]:
            st.warning(f"Percentile T1 ({r['suggested_t1']:.3f}) > suggested T2 ({r['suggested_t2']:.3f}) -- "
                       f"no clean single global cutoff for ArcFace on this data.")
        st.info("Copy the precision-method T1 and suggested_t2 into config.THRESHOLDS once you're "
                "happy with them -- the percentile T1 is shown for comparison but tends to be looser.")