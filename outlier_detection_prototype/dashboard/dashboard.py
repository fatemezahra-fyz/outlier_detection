import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="RAN Outlier Detection Dashboard", layout="wide")

st.title("📡 RAN Outlier Detection System")

# Use absolute paths based on this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'results.csv')
BASELINES_PATH = os.path.join(BASE_DIR, 'data', 'baselines.json')

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Results not found at {DATA_PATH}. Please run the pipeline first.")
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_baselines():
    if not os.path.exists(BASELINES_PATH):
        return None
    with open(BASELINES_PATH, 'r') as f:
        return json.load(f)

df = load_data()
baselines = load_baselines()

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Overview Map", "Outlier Table", "Sector Deep Dive", "Cluster Baselines"])

    with tab1:
        st.header("Network Anomaly Map")
        fig = px.scatter_mapbox(df,
                                lat="latitude",
                                lon="longitude",
                                color="outlier_score",
                                size="outlier_score",
                                color_continuous_scale=px.colors.sequential.Reds,
                                size_max=15,
                                zoom=11,
                                hover_name="sector_id",
                                hover_data=["cluster_id", "outlier_score", "top_contributing_kpi"],
                                mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Outlier Table")
        # Highlight outliers
        def highlight_outliers(row):
            return ['background-color: #ffcccc' if row.is_outlier else '' for _ in row]

        display_cols = ['sector_id', 'cluster_id', 'outlier_score', 'is_outlier', 'top_contributing_kpi', 'dl_throughput_mbps', 'rach_failure_rate_pct']
        st.dataframe(df[display_cols].style.apply(highlight_outliers, axis=1), use_container_width=True)

    with tab3:
        st.header("Sector Deep Dive")
        sector_id = st.selectbox("Select a sector", df['sector_id'].unique())
        sector_data = df[df['sector_id'] == sector_id].iloc[0]
        cluster_id = str(int(sector_data['cluster_id']))

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Outlier Score", f"{sector_data['outlier_score']:.3f}")
            st.write(f"**Cluster ID:** {cluster_id}")
            st.write(f"**Is Outlier:** {sector_data['is_outlier']}")
            st.write(f"**Top Contributor:** {sector_data['top_contributing_kpi']}")
            if sector_data['flag_silent_rach']:
                st.warning("⚠️ Silent RACH Failure Detected")
            if sector_data['neighbor_interference_surge']:
                st.warning("⚠️ Neighbor Interference Surge Detected")

        with col2:
            kpis = ['dl_throughput_mbps', 'ul_throughput_mbps', 'prb_utilization_pct', 'connected_users_mean', 'rach_failure_rate_pct', 'handover_success_rate_pct', 'ta_mean', 'dl_payload_gb']

            z_scores = [sector_data[f'{kpi}_zscore'] for kpi in kpis]

            fig_z = px.bar(x=kpis, y=z_scores, title=f"KPI Z-Scores for {sector_id} (Relative to Peer Group)")
            fig_z.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Outlier Threshold")
            fig_z.add_hline(y=-3, line_dash="dash", line_color="red")
            st.plotly_chart(fig_z, use_container_width=True)

    with tab4:
        st.header("Cluster Baselines")
        selected_cluster = st.selectbox("Select a cluster", sorted(df['cluster_id'].unique()))
        cluster_df = df[df['cluster_id'] == selected_cluster]

        kpi_to_plot = st.selectbox("Select KPI to visualize", kpis)

        fig_box = px.box(df, x="cluster_id", y=kpi_to_plot, color="cluster_id", title=f"{kpi_to_plot} distribution across clusters")
        st.plotly_chart(fig_box, use_container_width=True)

        if baselines and str(int(selected_cluster)) in baselines:
            st.write("### Baseline Statistics")
            st.json(baselines[str(int(selected_cluster))])
