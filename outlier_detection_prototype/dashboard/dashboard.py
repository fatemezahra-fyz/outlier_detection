import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Add src to path to import logger
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from logger import setup_logger

logger = setup_logger("dashboard")

# MTNIrancell Branding Colors
PRIMARY_COLOR = "#FFCC00" # Irancell Yellow/Orange
SECONDARY_COLOR = "#000000" # Black
ACCENT_COLOR = "#FF8C00" # Darker Orange

st.set_page_config(
    page_title="MTNIrancell RAN Anomaly Detection",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Injection
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: #ffffff;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: white;
        font-weight: 500;
    }}

    /* Headers */
    .main-header {{
        color: {ACCENT_COLOR};
        font-size: 36px;
        font-weight: 800;
        padding-bottom: 20px;
        border-bottom: 2px solid #eee;
        display: flex;
        align-items: center;
    }}

    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        color: {ACCENT_COLOR};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #fff3e0 !important;
        border-bottom: 3px solid {ACCENT_COLOR} !important;
    }}

    /* Alerts */
    .stAlert {{
        border-left: 5px solid {ACCENT_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.markdown(f'<div class="main-header">🟡 MTNIrancell RAN Anomaly Detection</div>', unsafe_allow_html=True)

# Absolute Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'results.csv')
BASELINES_PATH = os.path.join(BASE_DIR, 'data', 'baselines.json')

@st.cache_data
def load_data():
    logger.info(f"Loading results data from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Results file not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows of results data")
    return df

@st.cache_data
def load_baselines():
    logger.info(f"Loading baselines from {BASELINES_PATH}")
    if not os.path.exists(BASELINES_PATH):
        logger.error(f"Baselines file not found at {BASELINES_PATH}")
        return None
    with open(BASELINES_PATH, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded baselines for {len(data)} clusters")
    return data

df = load_data()
baselines = load_baselines()

if df is None:
    st.warning("⚠️ No result data found. Please run the ML Pipeline first: `python src/pipeline.py`")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/MTN_Group_Logo.svg/1024px-MTN_Group_Logo.svg.png", width=80)
    st.header("Network Summary")

    total_sectors = len(df)
    outliers_df = df[df['is_outlier']]
    outlier_count = len(outliers_df)
    outlier_rate = outlier_count / total_sectors

    col_s1, col_s2 = st.columns(2)
    st.metric("Total Sectors", total_sectors)
    st.metric("Detected Outliers", outlier_count, delta=f"{outlier_rate:.1%}", delta_color="inverse")

    st.markdown("---")
    st.subheader("Filter View")

    selected_vendors = st.multiselect("Vendor", options=df['vendor'].unique(), default=df['vendor'].unique())
    selected_layers = st.multiselect("Layer", options=df['layer'].unique(), default=df['layer'].unique())

    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Sector Deep Dive' tab to investigate specific cells flagged in red.")

# Filter Data
filtered_df = df[(df['vendor'].isin(selected_vendors)) & (df['layer'].isin(selected_layers))]

# --- MAIN CONTENT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Ahvaz Network Map",
    "📋 Outlier Analysis",
    "🔍 Sector Deep Dive",
    "📊 Peer Group Insights"
])

# Tab 1: Map
with tab1:
    st.header("Geospatial Anomaly Map - Ahvaz")

    color_map = px.colors.sequential.Oranges

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="outlier_score",
        size="outlier_score",
        color_continuous_scale=color_map,
        size_max=12,
        zoom=11.5,
        hover_name="sector_id",
        hover_data={
            "latitude": False,
            "longitude": False,
            "cluster_id": True,
            "outlier_score": ":.3f",
            "top_contributing_kpi": True
        },
        mapbox_style="carto-positron",
        title="Interactive Anomaly Heatmap"
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=600)
    st.plotly_chart(fig_map, use_container_width=True)

# Tab 2: Outlier Table
with tab2:
    st.header("Anomalous Sectors Detection")

    # Sort by outlier score
    table_df = filtered_df.sort_values("outlier_score", ascending=False)

    # KPIs to show in table
    display_cols = [
        'sector_id', 'vendor', 'layer', 'cluster_id',
        'outlier_score', 'is_outlier', 'top_contributing_kpi'
    ]

    def color_outliers(val):
        color = '#ffebee' if val == True else 'white'
        return f'background-color: {color}'

    st.dataframe(
        table_df[display_cols].style.map(color_outliers, subset=['is_outlier']),
        use_container_width=True,
        height=400
    )

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Outlier Distribution by Vendor")
        v_counts = outliers_df['vendor'].value_counts().reset_index()
        fig_pie = px.pie(v_counts, values='count', names='vendor',
                         color_discrete_sequence=[ACCENT_COLOR, "#FFD580", "#333333"])
        st.plotly_chart(fig_pie)

    with col_t2:
        st.subheader("Top Root Causes (KPIs)")
        k_counts = outliers_df['top_contributing_kpi'].value_counts().reset_index()
        fig_bar = px.bar(k_counts, x='top_contributing_kpi', y='count',
                         color_discrete_sequence=[ACCENT_COLOR])
        fig_bar.update_layout(xaxis_title="KPI", yaxis_title="Number of Outliers")
        st.plotly_chart(fig_bar)

# Tab 3: Deep Dive
with tab3:
    st.header("Sector Forensic Analysis")

    target_sector = st.selectbox("Select Sector for Detail Investigation", df['sector_id'].unique())
    s_row = df[df['sector_id'] == target_sector].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outlier Score", f"{s_row['outlier_score']:.3f}")
    c2.metric("Cluster ID", int(s_row['cluster_id']))

    impact_kpi = s_row['top_contributing_kpi'].replace('_', ' ').title()
    c3.metric("Top KPI Impact", impact_kpi)

    status_icon = "❌ CRITICAL" if s_row['is_outlier'] else "✅ HEALTHY"
    c4.metric("Status", status_icon)

    if s_row['is_outlier'] and 'top_features' in s_row and pd.notna(s_row['top_features']):
        st.info(f"**SHAP Root Cause Analysis:** {s_row['top_features']}")

    if s_row['flag_silent_rach'] or s_row['neighbor_interference_surge']:
        st.error("### 🚨 Forensic Alerts Detected")
        cols_a = st.columns(2)
        if s_row['flag_silent_rach']:
            cols_a[0].warning("**Silent RACH Failure**: High failure rate vs peer group.")
        if s_row['neighbor_interference_surge']:
            cols_a[1].warning("**Neighbor Interference**: Local handover performance issues.")

    st.markdown("---")

    # KPI Grid (Bar Charts with IQR)
    st.subheader("KPI forensic Analysis (Value vs. Peer Group)")

    from plotly.subplots import make_subplots
    cluster_id_str = str(int(s_row['cluster_id']))
    kpis = [
        'dl_throughput_mbps', 'ul_throughput_mbps', 'prb_utilization_pct',
        'connected_users_mean', 'rach_failure_rate_pct', 'handover_success_rate_pct',
        'ta_mean', 'dl_payload_gb'
    ]

    fig_grid = make_subplots(
        rows=2, cols=4,
        subplot_titles=[k.replace('_', ' ').upper() for k in kpis],
        vertical_spacing=0.15
    )

    for i, kpi in enumerate(kpis):
        row = (i // 4) + 1
        col = (i % 4) + 1
        b = baselines[cluster_id_str][kpi]

        # Sector vs Peer Median Bar
        fig_grid.add_trace(go.Bar(
            x=['Cell', 'Peer'],
            y=[s_row[kpi], b['median']],
            marker_color=[ACCENT_COLOR, 'lightgrey'],
            showlegend=False,
            text=[f"{s_row[kpi]:.1f}", f"{b['median']:.1f}"],
            textposition='auto',
        ), row=row, col=col)

        # IQR Band as Error Bar
        fig_grid.add_trace(go.Scatter(
            x=['Peer'],
            y=[b['median']],
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=1),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[b['q3'] - b['median']],
                arrayminus=[b['median'] - b['q1']],
                color='rgba(0,0,0,0.5)',
                thickness=10,
                width=15
            ),
            name='IQR Band',
            showlegend=False
        ), row=row, col=col)

    fig_grid.update_layout(height=600, margin=dict(t=50, b=50))
    st.plotly_chart(fig_grid, use_container_width=True)

    st.markdown("---")

    # Radar Chart
    st.subheader("Relative Performance Signature (Z-Scores)")
    kpis = [
        'dl_throughput_mbps', 'ul_throughput_mbps', 'prb_utilization_pct',
        'connected_users_mean', 'rach_failure_rate_pct', 'handover_success_rate_pct',
        'ta_mean', 'dl_payload_gb'
    ]
    z_vals = [s_row[f'{kpi}_zscore'] for kpi in kpis]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=z_vals,
        theta=[k.replace('_', ' ').upper() for k in kpis],
        fill='toself',
        name=f'Sector {target_sector}',
        line_color=ACCENT_COLOR,
        fillcolor='rgba(255, 140, 0, 0.3)'
    ))

    # Add peer group average (which is 0 in z-score space)
    fig_radar.add_trace(go.Scatterpolar(
        r=[0]*len(kpis),
        theta=[k.replace('_', ' ').upper() for k in kpis],
        fill=None,
        name='Peer Group Median',
        line_color='grey',
        line_dash='dash'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-4, 4], tickfont_size=10)
        ),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Tab 4: Baselines
with tab4:
    st.header("Peer Group Benchmarking")

    sel_cluster = st.selectbox("Select Peer Group (Cluster)", sorted(df['cluster_id'].unique()))

    kpi_choice = st.selectbox("Benchmark KPI", kpis, format_func=lambda x: x.replace('_', ' ').title())

    fig_box = px.box(
        df,
        x="cluster_id",
        y=kpi_choice,
        color="cluster_id",
        title=f"{kpi_choice.replace('_', ' ').title()} Distribution by Peer Group",
        color_discrete_sequence=px.colors.sequential.Oranges_r
    )

    # Add a point for the selected sector if it belongs to this cluster view
    st.plotly_chart(fig_box, use_container_width=True)

    if baselines:
        st.subheader(f"Group {sel_cluster} Statistics")
        cluster_key = str(int(sel_cluster))
        if cluster_key in baselines:
            stats_df = pd.DataFrame(baselines[cluster_key]).T
            st.table(stats_df)
        else:
            st.info("No baseline data available for this cluster ID.")
