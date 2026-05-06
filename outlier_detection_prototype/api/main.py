from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import os
from typing import List, Optional

app = FastAPI(title="RAN Outlier Detection API")

# Use absolute paths based on this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'results.csv')
BASELINES_PATH = os.path.join(BASE_DIR, 'data', 'baselines.json')

# In-memory storage for feedback
feedback_store = []

class Feedback(BaseModel):
    sector_id: str
    is_true_fault: bool
    comment: Optional[str] = None

def load_data():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=500, detail=f"Data not found at {DATA_PATH}. Run pipeline first.")
    return pd.read_csv(DATA_PATH)

def load_baselines():
    if not os.path.exists(BASELINES_PATH):
        raise HTTPException(status_code=500, detail="Baselines not found. Run pipeline first.")
    with open(BASELINES_PATH, 'r') as f:
        return json.load(f)

@app.get("/sectors")
def get_sectors():
    df = load_data()
    return df[['sector_id', 'outlier_score', 'is_outlier', 'cluster_id']].to_dict(orient='records')

@app.get("/sectors/{sector_id}")
def get_sector_detail(sector_id: str):
    df = load_data()
    sector = df[df['sector_id'] == sector_id]
    if sector.empty:
        raise HTTPException(status_code=404, detail="Sector not found")

    sector_info = sector.to_dict(orient='records')[0]
    cluster_id = str(int(sector_info['cluster_id']))

    baselines = load_baselines()
    peer_group_baseline = baselines.get(cluster_id)

    return {
        "sector_details": sector_info,
        "peer_group_baseline": peer_group_baseline
    }

@app.get("/outliers")
def get_outliers():
    df = load_data()
    outliers = df[df['is_outlier'] == True].sort_values('outlier_score', ascending=False)
    return outliers[['sector_id', 'cluster_id', 'outlier_score', 'top_contributing_kpi']].to_dict(orient='records')

@app.get("/clusters")
def get_clusters():
    df = load_data()
    summary = df.groupby('cluster_id').agg({
        'sector_id': 'count',
        'outlier_score': 'mean',
        'dl_throughput_mbps': 'median',
        'rach_failure_rate_pct': 'median'
    }).rename(columns={'sector_id': 'sector_count'}).reset_index()

    # Convert cluster_id to int for JSON serialization
    summary['cluster_id'] = summary['cluster_id'].astype(int)

    return summary.to_dict(orient='records')

@app.post("/feedback")
def post_feedback(fb: Feedback):
    feedback_store.append(fb.dict())
    return {"message": "Feedback received", "count": len(feedback_store)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
