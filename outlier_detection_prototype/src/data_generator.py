import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_sectors=500, seed=42):
    np.random.seed(seed)

    # Configuration (CM) features
    vendors = ['Ericsson', 'Huawei', 'Nokia']
    layers = ['1800MHz', '2100MHz', '2600MHz']
    bandwidths = [10, 15, 20]
    mimo_configs = ['2x2', '4x4', '8x8']

    sector_ids = [f"SEC_{i:03d}" for i in range(num_sectors)]

    data = {
        'sector_id': sector_ids,
        'vendor': np.random.choice(vendors, num_sectors),
        'layer': np.random.choice(layers, num_sectors),
        'bandwidth_mhz': np.random.choice(bandwidths, num_sectors),
        'mimo_config': np.random.choice(mimo_configs, num_sectors),
        'antenna_tilt_mechanical': np.random.uniform(0, 10, num_sectors),
        'antenna_tilt_electrical': np.random.uniform(0, 15, num_sectors),
        'transmit_power_dbm': np.random.uniform(30, 46, num_sectors),
        'azimuth': np.random.uniform(0, 360, num_sectors),
        'latitude': np.random.uniform(31.2, 31.4, num_sectors), # Ahvaz area
        'longitude': np.random.uniform(48.6, 48.8, num_sectors)
    }

    df = pd.DataFrame(data)

    # Performance (PM) KPIs
    df['dl_throughput_mbps'] = np.random.normal(50, 15, num_sectors).clip(5, 150)
    df['ul_throughput_mbps'] = np.random.normal(10, 3, num_sectors).clip(1, 30)
    df['prb_utilization_pct'] = np.random.uniform(20, 80, num_sectors)
    df['connected_users_mean'] = np.random.poisson(30, num_sectors)
    df['rach_failure_rate_pct'] = np.random.exponential(1, num_sectors).clip(0, 10)
    df['handover_success_rate_pct'] = np.random.uniform(95, 100, num_sectors)
    df['ta_mean'] = np.random.uniform(1, 10, num_sectors)
    df['dl_payload_gb'] = np.random.normal(100, 30, num_sectors).clip(10, 500)

    df.loc[df['layer'] == '2600MHz', 'dl_throughput_mbps'] *= 1.5
    df.loc[df['layer'] == '2600MHz', 'ta_mean'] *= 0.7

    num_anomalies = int(num_sectors * 0.05)
    anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['low_throughput', 'high_failure', 'high_interference'])
        if anomaly_type == 'low_throughput':
            df.loc[idx, 'dl_throughput_mbps'] *= 0.1
            df.loc[idx, 'ul_throughput_mbps'] *= 0.1
        elif anomaly_type == 'high_failure':
            df.loc[idx, 'rach_failure_rate_pct'] = np.random.uniform(20, 50)
            df.loc[idx, 'handover_success_rate_pct'] = np.random.uniform(70, 85)
        elif anomaly_type == 'high_interference':
            df.loc[idx, 'prb_utilization_pct'] = np.random.uniform(90, 100)
            df.loc[idx, 'dl_throughput_mbps'] *= 0.3

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    # Use relative path from the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'data', 'synthetic_sectors.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sectors and saved to {output_path}")
