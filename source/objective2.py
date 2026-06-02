"""
Objective 2 — Risk Segmentation (Clustering)
==============================================
Applies K-Means clustering on five financial behaviour columns to segment
credit card holders into three distinct risk groups.

Output (saved to output/):
  • risk_segments_profile.csv
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Main callable function
# ---------------------------------------------------------------------------
def run_clustering(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Segment customers into 3 risk clusters based on financial behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed Taiwan Credit data (output of Objective 1).
    output_dir : str
        Directory path where the profile CSV will be saved.

    Returns
    -------
    pd.DataFrame
        The DataFrame augmented with a 'Risk_Segment' column.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = df.copy()

    # ── 1. Select financial behaviour columns ───────────────────────────
    cluster_cols = ["LIMIT_BAL", "PAY_0", "BILL_AMT1", "PAY_AMT1", "class"]
    cluster_data = data[cluster_cols].copy()

    # ── 2. Standardise features ─────────────────────────────────────────
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)

    # ── 3. K-Means Clustering (k = 3) ──────────────────────────────────
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    data["Risk_Segment"] = kmeans.fit_predict(scaled_features)

    # ── 4. Build profile summary ────────────────────────────────────────
    profile = (
        data.groupby("Risk_Segment")[cluster_cols]
        .mean()
        .round(2)
    )
    profile["Customer_Count"] = data.groupby("Risk_Segment")["Risk_Segment"].count()

    # ── 5. Export ───────────────────────────────────────────────────────
    path = os.path.join(output_dir, "risk_segments_profile.csv")
    profile.to_csv(path)

    print("\n" + "=" * 60)
    print("  OBJECTIVE 2 — Risk Segmentation Results")
    print("=" * 60)
    print(profile.to_string())
    print("-" * 60)
    print("  [OK] Saved: {}".format(path))
    print("=" * 60)

    return data
