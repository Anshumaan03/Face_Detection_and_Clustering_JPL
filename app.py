import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Face Clustering Dashboard",
    layout="wide"
)

st.title("Face Clustering Dashboard")

with open("clustering_results.json") as f:
    results = json.load(f)

best = results["best_params"]

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric("NMI", f"{best['nmi']:.3f}")
col2.metric("ARI", f"{best['ari']:.3f}")
col3.metric("Purity", f"{best['purity']:.3f}")
col4.metric("Clusters", best["n_clusters"])
col5.metric("Noise %", f"{best['noise_pct']:.2f}")