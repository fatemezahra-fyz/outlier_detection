import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
from geopy.distance import geodesic

def compute_outlier_scores(df, kpis, baselines):
    """
    Step 5: Outlier Detection
    """
    df = df.copy()

    # Per-KPI z-score relative to peer group
    kpi_z_cols = []
    for kpi in kpis:
        z_col = f'{kpi}_zscore'
        kpi_z_cols.append(z_col)

        def calc_z(row):
            cluster_id = int(row['cluster_id'])
            baseline = baselines[cluster_id][kpi]
            # Robust z-score using Median and IQR
            # z = (x - median) / (1.35 * IQR) - but the prompt just says median and IQR.
            # I'll use (x - median) / IQR (handling IQR=0)
            iqr = baseline['iqr']
            if iqr == 0:
                return 0.0
            return (row[kpi] - baseline['median']) / iqr

        df[z_col] = df.apply(calc_z, axis=1)

    # Composite outlier score (0 to 1)
    # 1. Max absolute z-score
    df['max_abs_z'] = df[kpi_z_cols].abs().max(axis=1)

    # 2. Normalize using sigmoid
    # Score = 1 / (1 + exp(- (max_abs_z - threshold) * scaling))
    # Let's use a simpler min-max or sigmoid.
    # Sigmoid centered at 3 (typical outlier threshold)
    df['outlier_score'] = 1 / (1 + np.exp(-(df['max_abs_z'] - 3)))

    df['is_outlier'] = df['outlier_score'] > 0.7

    # Top contributing KPI
    def get_top_kpi(row):
        z_scores = {kpi: abs(row[f'{kpi}_zscore']) for kpi in kpis}
        return max(z_scores, key=z_scores.get)

    df['top_contributing_kpi'] = df.apply(get_top_kpi, axis=1)

    # Special Flags
    # Silent RACH failure
    def flag_silent_rach(row):
        cluster_id = int(row['cluster_id'])
        baseline = baselines[cluster_id]['rach_failure_rate_pct']
        threshold = baseline['median'] + 2.5 * baseline['iqr']
        return row['rach_failure_rate_pct'] > threshold

    df['flag_silent_rach'] = df.apply(flag_silent_rach, axis=1)

    # Neighbor interference surge (within 1km)
    # sector whose neighbors (within 1km) have unusually high handover failure rates directed at it
    # Since we don't have "directed" handover failures in synthetic data,
    # we'll use a proxy: neighbors have low handover success rate.

    df['neighbor_interference_surge'] = False
    coords = df[['latitude', 'longitude']].values
    for i in range(len(df)):
        neighbors_idx = []
        for j in range(len(df)):
            if i == j: continue
            dist = geodesic(coords[i], coords[j]).km
            if dist <= 1.0:
                neighbors_idx.append(j)

        if neighbors_idx:
            neighbor_homsr = df.iloc[neighbors_idx]['handover_success_rate_pct']
            # If average neighbor HOMSR is low (e.g., < 90%)
            if neighbor_homsr.mean() < 95:
                df.at[i, 'neighbor_interference_surge'] = True

    return df

def explain_outliers(df, features_df):
    """
    Step 6: SHAP Explainability
    """
    top_outliers_idx = df.sort_values('outlier_score', ascending=False).head(10).index

    # Train IsolationForest on all features for explainability
    model = IsolationForest(random_state=42)
    model.fit(features_df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df.loc[top_outliers_idx])

    explanations = []
    feature_names = features_df.columns.tolist()

    print("\n--- SHAP Feature Importance for Top 10 Outliers ---")
    for i, idx in enumerate(top_outliers_idx):
        # shap_values is an array [num_samples, num_features]
        sv = shap_values[i]
        top_3_indices = np.argsort(np.abs(sv))[-3:][::-1]
        top_3_features = [f"{feature_names[j]} ({sv[j]:.2f})" for j in top_3_indices]
        sector_id = df.loc[idx, 'sector_id']
        print(f"Sector {sector_id}: {', '.join(top_3_features)}")
        explanations.append({
            'sector_id': sector_id,
            'top_features': top_3_features
        })

    return explanations
