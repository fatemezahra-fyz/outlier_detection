import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
from logger import setup_logger

logger = setup_logger("feature_engineering")

def engineer_features(df):
    """
    Step 2: Feature Engineering Pipeline
    """
    logger.info("Starting feature engineering...")
    df = df.copy()

    # 1. Derived features
    df['total_tilt'] = df['antenna_tilt_mechanical'] + df['antenna_tilt_electrical']
    df['electrical_tilt_ratio'] = df['antenna_tilt_electrical'] / df['total_tilt'].replace(0, np.nan)
    df['electrical_tilt_ratio'] = df['electrical_tilt_ratio'].fillna(0)

    # transmit_power_dbm is in dBm, let's convert to Watts for density calculation or just use as is?
    # Spec says power_density_per_mhz = transmit_power / bandwidth
    # I will assume transmit_power in Watts for the ratio to be more meaningful,
    # but since it's synthetic and will be normalized, dBm/MHz is also fine.
    # Let's use 10^((dbm-30)/10) to get Watts.
    df['transmit_power_watts'] = 10**((df['transmit_power_dbm'] - 30) / 10)
    df['power_density_per_mhz'] = df['transmit_power_watts'] / df['bandwidth_mhz']

    # numeric_frequency_mhz
    df['numeric_frequency_mhz'] = df['layer'].str.extract(r'(\d+)').astype(float)

    # Geo features
    coords = df[['latitude', 'longitude']].values

    # Distance to nearest site (excluding self)
    # Using a simple approximation or geodesic? Geopy is requested.
    # For 500 points, we can do it efficiently.

    distances = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist = geodesic(coords[i], coords[j]).km
            distances[i, j] = dist
            distances[j, i] = dist

    # Set diagonal to infinity to find nearest neighbor
    np.fill_diagonal(distances, np.inf)

    df['distance_to_nearest_site'] = np.min(distances, axis=1)
    df['local_site_density'] = np.sum(distances <= 2.0, axis=1)

    logger.info("Feature engineering complete.")
    return df

def preprocess_features(df):
    """
    Apply preprocessing: Z-score, One-hot, Imputation
    """
    logger.info("Starting feature preprocessing...")
    df = df.copy()

    numeric_features = [
        'bandwidth_mhz', 'antenna_tilt_mechanical', 'antenna_tilt_electrical',
        'transmit_power_dbm', 'azimuth', 'latitude', 'longitude',
        'total_tilt', 'electrical_tilt_ratio', 'power_density_per_mhz',
        'numeric_frequency_mhz', 'distance_to_nearest_site', 'local_site_density'
    ]

    categorical_features = ['vendor', 'layer', 'mimo_config']

    # Imputation
    imputer = SimpleImputer(strategy='median')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])

    # Normalization
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]),
                             columns=numeric_features,
                             index=df.index)

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df.index)

    # Combine
    features_final = pd.concat([df_scaled, df_encoded], axis=1)

    logger.info(f"Preprocessing complete. Final feature set shape: {features_final.shape}")
    return features_final, scaler, encoder, imputer

if __name__ == "__main__":
    import os
    data_path = os.path.join('outlier_detection_prototype', 'data', 'synthetic_sectors.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df_engineered = engineer_features(df)
        features_final, _, _, _ = preprocess_features(df_engineered)
        print("Feature engineering successful.")
        print(f"Original shape: {df.shape}")
        print(f"Engineered features shape: {features_final.shape}")
        print(f"Columns: {features_final.columns.tolist()}")
    else:
        print("Synthetic data not found. Run data_generator.py first.")
