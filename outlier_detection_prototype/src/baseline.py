import pandas as pd
from logger import setup_logger

logger = setup_logger("baseline")

def compute_peer_group_baselines(df, kpis):
    """
    Step 4: Baseline Modeling Per Peer Group
    For each peer group, compute per-KPI baseline statistics:
    - Median, Q1, Q3, IQR
    """
    logger.info("Computing peer group baselines...")
    baselines = {}

    unique_clusters = df['cluster_id'].unique()
    logger.info(f"Found {len(unique_clusters)} unique clusters.")

    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster_id'] == cluster_id]
        cluster_baselines = {}

        for kpi in kpis:
            kpi_data = cluster_df[kpi]
            median = kpi_data.median()
            q1 = kpi_data.quantile(0.25)
            q3 = kpi_data.quantile(0.75)
            iqr = q3 - q1

            cluster_baselines[kpi] = {
                'median': float(median),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }

        baselines[int(cluster_id)] = cluster_baselines

    logger.info("Baseline computation complete.")
    return baselines

if __name__ == "__main__":
    # Test with sample data
    df_test = pd.DataFrame({
        'cluster_id': [0, 0, 0, 1, 1, 1],
        'kpi1': [10, 12, 11, 100, 110, 105],
        'kpi2': [5, 6, 5.5, 50, 60, 55]
    })
    kpis = ['kpi1', 'kpi2']
    baselines = compute_peer_group_baselines(df_test, kpis)
    print("Computed Baselines:")
    import json
    print(json.dumps(baselines, indent=2))
