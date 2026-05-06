# RAN Outlier Detection System Prototype

This prototype implements an Intelligent Network Anomaly Detection system for Radio Access Network (RAN) sectors. It uses peer-grouping and robust statistical baselines to detect performance outliers.

## Features
- **Synthetic Data Generation**: Simulates 500 RAN sectors with CM and PM data.
- **Peer Group Clustering**: Automatically groups similar sectors using K-Means/GMM.
- **Robust Outlier Scoring**: Detects deviations using median and IQR per peer group.
- **SHAP Explainability**: Provides feature importance for top outliers.
- **FastAPI**: REST API for integration.
- **Streamlit**: Interactive dashboard for visualization.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the ML Pipeline
This script generates synthetic data, performs clustering, computes baselines, and flags outliers.
```bash
python src/pipeline.py
```
Results will be saved to `data/results.csv` and `data/baselines.json`.

### 2. Start the API
```bash
uvicorn api.main:app --reload
```
The API will be available at `http://localhost:8000`.

**Example API Commands:**
- Get all sectors: `curl http://localhost:8000/sectors`
- Get outliers: `curl http://localhost:8000/outliers`
- Get cluster summary: `curl http://localhost:8000/clusters`
- Submit feedback:
  ```bash
  curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d '{"sector_id": "SEC_001", "is_true_fault": true, "comment": "Verified anomaly"}'
  ```

### 3. Launch the Dashboard
```bash
streamlit run dashboard/dashboard.py
```
The dashboard provides a spatial map, outlier tables, and deep-dive visualizations.

## Project Structure
- `src/`: Core logic (data generation, feature engineering, clustering, etc.)
- `api/`: FastAPI application.
- `dashboard/`: Streamlit dashboard.
- `data/`: Generated datasets and model outputs.
