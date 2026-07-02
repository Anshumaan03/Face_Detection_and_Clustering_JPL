import os
import cv2
import json
import numpy as np
import pandas as pd
import mysql.connector
import streamlit as st
from sklearn.metrics.pairwise import pairwise_distances
from pipeline import DB_CONFIG, DATASET_DIR, get_affine_aligned_face, get_arcface_embedding, yolo_model

# 📏 Three-Tier Recommendation Cut-off Distances (Cosine Space)
T1_AUTO_MERGE = 0.18      
T2_USER_PROMPT = 0.40     

st.set_page_config(page_title="Engine Cluster UI Dashboard", layout="wide")
st.title("📸 ArcFace Production Multi-Face Clustering Dashboard")
st.markdown("---")

def pull_database_records():
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT id, file_name, identity_label, cluster_id FROM face_embeddings", conn)
    conn.close()
    return df

db_df = pull_database_records()

def get_unit_centroid(embeddings_list):
    """Finds the mean vector and normalizes it back onto the unit sphere."""
    vectors = np.array(embeddings_list, dtype=np.float32)
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean

def evaluate_recommendation_tier(new_embeddings):
    """Processes newly uploaded face batches against your database centroids using the 3-tier threshold strategy."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT cluster_id, embedding_json FROM face_embeddings WHERE cluster_id != -1")
    rows = cursor.fetchall()
    
    if not rows:
        cursor.close()
        conn.close()
        return {"action": "NEW_CLUSTER", "target": None, "dist": 1.0}
        
    clusters_map = {}
    for cid, json_str in rows:
        clusters_map.setdefault(cid, []).append(json.loads(json_str))
        
    existing_ids = list(clusters_map.keys())
    existing_centroids = [get_unit_centroid(clusters_map[cid]) for cid in existing_ids]
    new_centroid = get_unit_centroid(new_embeddings).reshape(1, -1)
    
    # Measure the distance from the new batch to existing centroids
    distances = pairwise_distances(np.array(existing_centroids), new_centroid, metric='cosine').flatten()
    closest_idx = np.argmin(distances)
    min_dist = distances[closest_idx]
    target_cluster = existing_ids[closest_idx]
    
    cursor.close()
    conn.close()
    
    if min_dist <= T1_AUTO_MERGE:
        return {"action": "AUTO_MERGE", "target": int(target_cluster), "dist": float(min_dist)}
    elif min_dist <= T2_USER_PROMPT:
        return {"action": "USER_PROMPT", "target": int(target_cluster), "dist": float(min_dist)}
    else:
        return {"action": "NEW_CLUSTER", "target": None, "dist": float(min_dist)}

# ==========================================
# 📥 STREAMLIT SIDEBAR: LIVE STREAM UPLOADER
# ==========================================
with st.sidebar:
    st.header("📥 Ingest New Batch")
    uploaded_files = st.file_uploader("Upload new photos to cluster:", accept_multiple_files=True, type=['jpg','png','jpeg'])
    
    if uploaded_files and st.button("🧠 Run Real-Time Stream matching"):
        stream_embeddings = []
        stream_references = []
        
        stream_save_path = os.path.join(DATASET_DIR, "Live_Upload_Stream")
        os.makedirs(stream_save_path, exist_ok=True)
        
        for u_file in uploaded_files:
            file_bytes = np.frombuffer(u_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None: continue
            
            disk_path = os.path.join(stream_save_path, u_file.name)
            cv2.imwrite(disk_path, img)
            
            res = yolo_model(img, verbose=False)
            if res and res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
                kpts = res[0].keypoints.data[0].cpu().numpy()
                aligned = get_affine_aligned_face(img, kpts)
                if aligned is not None:
                    vec = get_arcface_embedding(aligned)
                    stream_embeddings.append(vec)
                    stream_references.append((u_file.name, vec.tolist()))
                    
        if stream_embeddings:
            verdict = evaluate_recommendation_tier(stream_embeddings)
            st.session_state["active_verdict"] = verdict
            st.session_state["active_references"] = stream_references
            st.success(f"Stream analysis completed: {verdict['action']}")
        else:
            st.error("Could not extract face geometries from the uploaded batch.")

# ==========================================
# 🧠 INTERACTIVE INTERACTION PORTS
# ==========================================
if "active_verdict" in st.session_state:
    st.header("⚡ Recommendation Engine Active Decision Gate")
    v = st.session_state["active_verdict"]
    r = st.session_state["active_references"]
    
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    if v["action"] == "AUTO_MERGE":
        st.success(f"🤖 **Auto-Merge Triggered:** This batch closely matches existing **Cluster ID #{v['target_cluster']}** (Distance: {v['dist']:.3f}). Appending vectors directly.")
        for name, vector in r:
            cursor.execute("""
                INSERT INTO face_embeddings (file_name, identity_label, embedding_json, cluster_id) 
                VALUES (%s, 'Live_Upload_Stream', %s, %s)
            """, (name, json.dumps(vector), v["target_cluster"]))
        conn.commit()
        del st.session_state["active_verdict"]
        st.rerun()
        
    elif v["action"] == "USER_PROMPT":
        st.warning(f"❓ **Verification Prompt:** The uploaded faces are moderately close to **Cluster ID #{v['target_cluster']}** (Distance: {v['dist']:.3f}). Is this the same person?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Merge and Update Cluster Mapping", use_container_width=True):
                for name, vector in r:
                    cursor.execute("""
                        INSERT INTO face_embeddings (file_name, identity_label, embedding_json, cluster_id) 
                        VALUES (%s, 'Live_Upload_Stream', %s, %s)
                    """, (name, json.dumps(vector), v["target_cluster"]))
                conn.commit()
                st.success("Database records linked and cluster merged successfully!")
                del st.session_state["active_verdict"]
                st.rerun()
        with col2:
            if st.button("❌ No, Keep Separate and Create New Cluster ID", use_container_width=True):
                cursor.execute("SELECT MAX(cluster_id) FROM face_embeddings")
                max_cid = cursor.fetchone()[0]
                new_id = (max_cid + 1) if max_cid and max_cid >= 0 else 0
                for name, vector in r:
                    cursor.execute("""
                        INSERT INTO face_embeddings (file_name, identity_label, embedding_json, cluster_id) 
                        VALUES (%s, 'Live_Upload_Stream', %s, %s)
                    """, (name, json.dumps(vector), new_id))
                conn.commit()
                st.info(f"Isolating batch under a new unique profile: **Cluster #{new_id}**")
                del st.session_state["active_verdict"]
                st.rerun()
                
    elif v["action"] == "NEW_CLUSTER":
        st.info("🌟 **Distinct Profile Verified:** Distance exceeds thresholds. Creating a separate cluster identity profile.")
        cursor.execute("SELECT MAX(cluster_id) FROM face_embeddings")
        max_cid = cursor.fetchone()[0]
        new_id = (max_cid + 1) if max_cid and max_cid >= 0 else 0
        for name, vector in r:
            cursor.execute("""
                INSERT INTO face_embeddings (file_name, identity_label, embedding_json, cluster_id) 
                VALUES (%s, 'Live_Upload_Stream', %s, %s)
            """, (name, json.dumps(vector), new_id))
        conn.commit()
        del st.session_state["active_verdict"]
        st.rerun()
        
    cursor.close()
    conn.close()
    st.markdown("---")

# ==========================================
# 📑 UI TABS FOR VISUAL GALLERY STRUCTURES
# ==========================================
tab1, tab2 = st.tabs(["📁 Source Directories View", "🧠 Model Generated Identity Clusters"])

with tab1:
    st.subheader("📁 Browse Aligned Files by Raw Dataset Folder")
    available_folders = sorted(db_df['identity_label'].unique())
    selected_folder = st.selectbox("Choose raw dataset folder folder path:", available_folders)
    
    filtered_df = db_df[db_df['identity_label'] == selected_folder]
    grid = st.columns(4)
    for index, row in filtered_df.reset_index().iterrows():
        img_path = os.path.join(DATASET_DIR, selected_folder, row['file_name'])
        with grid[index % 4]:
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
                st.caption(f"📄 {row['file_name']} | Assigned Cluster: **#{row['cluster_id']}**")

with tab2:
    st.subheader("🧠 Browse Images Grouped by Calculated Cluster ID")
    available_clusters = sorted(db_df['cluster_id'].unique())
    cluster_labels = [f"Identity Group Profile #{c}" if c != -1 else "⚠️ System Flagged Noise Outliers (-1)" for c in available_clusters]
    cluster_mapping = dict(zip(cluster_labels, available_clusters))
    
    selected_label = st.selectbox("Choose generated cluster group:", cluster_labels)
    target_cluster_id = cluster_mapping[selected_label]
    
    cluster_df = db_df[db_df['cluster_id'] == target_cluster_id]
    cluster_grid = st.columns(5)
    for index, row in cluster_df.reset_index().iterrows():
        img_path = os.path.join(DATASET_DIR, row['identity_label'], row['file_name'])
        with cluster_grid[index % 5]:
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
                st.caption(f"📁 Source Folder: **{row['identity_label']}**\n\n📄 File: `{row['file_name']}`")