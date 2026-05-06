import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_clustering_comparison(features_df):
    """
    Step 3: Peer Group Clustering (Similarity Modeling)
    """
    results = []

    # 1. K-Means
    for k in [5, 8, 10]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features_df)
        results.append({
            'model': f'K-Means (k={k})',
            'labels': labels,
            'silhouette': silhouette_score(features_df, labels),
            'db_index': davies_bouldin_score(features_df, labels)
        })

    # 2. HDBSCAN
    model_hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
    labels_hdbscan = model_hdbscan.fit_predict(features_df)
    # HDBSCAN can produce -1 for noise, silhouette score needs at least 2 clusters (excluding noise)
    unique_labels = set(labels_hdbscan) - {-1}
    if len(unique_labels) >= 2:
        # For evaluation, we might need to filter out noise or treat it as a cluster
        # Let's filter for score calculation
        mask = labels_hdbscan != -1
        if mask.sum() > 0:
            results.append({
                'model': 'HDBSCAN',
                'labels': labels_hdbscan,
                'silhouette': silhouette_score(features_df[mask], labels_hdbscan[mask]),
                'db_index': davies_bouldin_score(features_df[mask], labels_hdbscan[mask])
            })

    # 3. GMM
    for n in [5, 8, 10]:
        model = GaussianMixture(n_components=n, random_state=42)
        labels = model.fit_predict(features_df)
        results.append({
            'model': f'GMM (n={n})',
            'labels': labels,
            'silhouette': silhouette_score(features_df, labels),
            'db_index': davies_bouldin_score(features_df, labels)
        })

    results_df = pd.DataFrame(results)
    return results_df

def select_best_clustering(results_df):
    # Higher silhouette and lower DB index is better.
    # Let's use a simple heuristic: highest silhouette.
    best_row = results_df.loc[results_df['silhouette'].idxmax()]
    return best_row

if __name__ == "__main__":
    from data_generator import generate_synthetic_data
    from feature_engineering import engineer_features, preprocess_features

    df = generate_synthetic_data()
    df_eng = engineer_features(df)
    features, _, _, _ = preprocess_features(df_eng)

    comparison = run_clustering_comparison(features)
    print("Clustering Comparison:")
    print(comparison[['model', 'silhouette', 'db_index']])

    best = select_best_clustering(comparison)
    print(f"\nBest Model: {best['model']}")

    labels = best['labels']
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Summary:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} sectors")
