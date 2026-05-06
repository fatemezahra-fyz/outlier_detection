import pandas as pd
import os
import json
from data_generator import generate_synthetic_data
from feature_engineering import engineer_features, preprocess_features
from clustering import run_clustering_comparison, select_best_clustering
from baseline import compute_peer_group_baselines
from outlier_detection import compute_outlier_scores, explain_outliers

def run_pipeline():
    print("Starting pipeline...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1. Data Generation
    df = generate_synthetic_data()
    print("Data generated.")

    # 2. Feature Engineering
    df_eng = engineer_features(df)
    features, scaler, encoder, imputer = preprocess_features(df_eng)
    print("Features engineered and preprocessed.")

    # 3. Clustering
    clustering_results = run_clustering_comparison(features)
    best_clustering = select_best_clustering(clustering_results)
    df['cluster_id'] = best_clustering['labels']
    print(f"Clustering complete. Best model: {best_clustering['model']}")

    # 4. Baselines
    kpis = [
        'dl_throughput_mbps', 'ul_throughput_mbps', 'prb_utilization_pct',
        'connected_users_mean', 'rach_failure_rate_pct', 'handover_success_rate_pct',
        'ta_mean', 'dl_payload_gb'
    ]
    baselines = compute_peer_group_baselines(df, kpis)
    print("Baselines computed.")

    # 5. Outlier Detection
    df_results = compute_outlier_scores(df, kpis, baselines)
    print("Outlier scores computed.")

    # 6. SHAP Explainability
    explanations = explain_outliers(df_results, features)
    print("SHAP explanations generated for top 10 outliers.")

    # Save results
    df_results.to_csv(os.path.join(data_dir, 'results.csv'), index=False)

    # Save baselines for API/Dashboard use
    with open(os.path.join(data_dir, 'baselines.json'), 'w') as f:
        json.dump(baselines, f)

    print(f"Pipeline complete. Results saved to {data_dir}")

if __name__ == "__main__":
    run_pipeline()
