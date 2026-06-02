"""
Objective 3 — Data Warehouse + OLAP Cube
==========================================
Creates human-readable dimension columns, bins AGE and LIMIT_BAL, and
performs a multi-dimensional GroupBy OLAP operation to compute key business
measures.

Output (saved to output/):
  • olap_cube_report.csv
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dimension mapping helpers
# ---------------------------------------------------------------------------
def _map_gender(code):
    mapping = {1: "Male", 2: "Female"}
    return mapping.get(code, "Unknown")


def _map_education(code):
    mapping = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    return mapping.get(code, "Unknown")


def _map_marriage(code):
    mapping = {1: "Married", 2: "Single", 3: "Others"}
    return mapping.get(code, "Unknown")


# ---------------------------------------------------------------------------
# Main callable function
# ---------------------------------------------------------------------------
def run_olap(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Build dimension columns and produce a multi-dimensional OLAP cube report.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed Taiwan Credit data.
    output_dir : str
        Directory path where the OLAP report will be saved.

    Returns
    -------
    pd.DataFrame
        The OLAP cube summary table.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = df.copy()

    # ── 1. Create Dimension Columns ─────────────────────────────────────

    # AGE → AGE_GROUP
    bins_age = [0, 25, 35, 45, 55, np.inf]
    labels_age = ["Under 25", "25-35", "35-45", "45-55", "Above 55"]
    data["AGE_GROUP"] = pd.cut(data["AGE"], bins=bins_age, labels=labels_age, right=False)

    # LIMIT_BAL → CREDIT_TIER (quantile-based, 3 tiers)
    data["CREDIT_TIER"] = pd.qcut(
        data["LIMIT_BAL"], q=3, labels=["Low", "Medium", "High"]
    )

    # SEX → GENDER
    data["GENDER"] = data["SEX"].apply(_map_gender)

    # EDUCATION → EDUCATION_LEVEL
    data["EDUCATION_LEVEL"] = data["EDUCATION"].apply(_map_education)

    # MARRIAGE → MARITAL_STATUS
    data["MARITAL_STATUS"] = data["MARRIAGE"].apply(_map_marriage)

    # ── 2. Define dimensions and measures ───────────────────────────────
    dimensions = ["AGE_GROUP", "CREDIT_TIER", "GENDER", "EDUCATION_LEVEL", "MARITAL_STATUS"]

    # ── 3. Multi-dimensional OLAP GroupBy ───────────────────────────────
    olap_cube = (
        data.groupby(dimensions, observed=False)
        .agg(
            default_rate=("class", "mean"),
            average_bill=("BILL_AMT1", "mean"),
            average_payment=("PAY_AMT1", "mean"),
            customer_count=("class", "count"),
        )
        .reset_index()
        .round(4)
    )

    # ── 4. Export ───────────────────────────────────────────────────────
    path = os.path.join(output_dir, "olap_cube_report.csv")
    olap_cube.to_csv(path, index=False)

    print("\n" + "=" * 60)
    print("  OBJECTIVE 3 — OLAP Cube Report")
    print("=" * 60)
    print(f"  Dimensions  : {dimensions}")
    print(f"  Total Cells : {len(olap_cube)}")
    print(f"  Non-Empty   : {(olap_cube['customer_count'] > 0).sum()}")
    print("-" * 60)
    print(olap_cube.head(15).to_string(index=False))
    print("  ...")
    print("-" * 60)
    print("  [OK] Saved: {}".format(path))
    print("=" * 60)

    return olap_cube
