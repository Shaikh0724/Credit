"""
CCP - Complete CRISP Pipeline Orchestrator
============================================
Master controller that sequentially executes all three analytical objectives
for the Taiwan Credit Card Default Analytics project.

Usage:
    cd C:\\Users\\DANISH\\Desktop\\Credit
    python model/ccp.py
"""

import os
import sys
import time
import io
import pandas as pd

# ---------------------------------------------------------------------------
# Force UTF-8 stdout so box-drawing and emoji chars render on Windows
# ---------------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup - ensure cross-directory imports work regardless of CWD
# ---------------------------------------------------------------------------
# Resolve the project root (one level up from this script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Add project root to sys.path so `source.*` modules can be imported
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Key paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "TaiwanData.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ---------------------------------------------------------------------------
# Imports from the source package (after path is configured)
# ---------------------------------------------------------------------------
from source.objective1 import run_classification
from source.objective2 import run_clustering
from source.objective3 import run_olap


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _banner(text):
    """Print a prominent section banner to the terminal."""
    width = 70
    print("\n")
    print("+" + "=" * width + "+")
    print("|" + text.center(width) + "|")
    print("+" + "=" * width + "+")


def _milestone(msg, start):
    elapsed = time.time() - start
    print("\n  [OK] {} completed in {:.2f}s".format(msg, elapsed))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    pipeline_start = time.time()

    _banner("CREDIT CARD DEFAULT ANALYTICS -- CRISP PIPELINE")
    print("  Project Root : {}".format(PROJECT_ROOT))
    print("  Data File    : {}".format(DATA_PATH))
    print("  Output Dir   : {}".format(OUTPUT_DIR))

    # -- 0. Load Data --------------------------------------------------------
    _banner("PHASE 0 -- Loading Dataset")
    if not os.path.isfile(DATA_PATH):
        print("  [ERROR] Dataset not found at {}".format(DATA_PATH))
        sys.exit(1)

    df_raw = pd.read_csv(DATA_PATH)
    print("  [OK] Loaded {:,} rows x {} columns".format(df_raw.shape[0], df_raw.shape[1]))
    print("  [OK] Columns: {}".format(list(df_raw.columns)))

    # Drop the ID column if present - it is not a feature
    if "ID" in df_raw.columns:
        df_raw.drop(columns=["ID"], inplace=True)
        print("  [OK] Dropped 'ID' column (non-feature)")

    # -- 1. Objective 1 -- Default Prediction (Classification) ---------------
    _banner("PHASE 1 -- Objective 1: Default Prediction")
    t1 = time.time()
    df_clean = run_classification(df_raw, OUTPUT_DIR)
    _milestone("Objective 1 (Classification)", t1)

    # -- 2. Objective 2 -- Risk Segmentation (Clustering) --------------------
    _banner("PHASE 2 -- Objective 2: Risk Segmentation")
    t2 = time.time()
    df_segmented = run_clustering(df_clean, OUTPUT_DIR)
    _milestone("Objective 2 (Clustering)", t2)

    # -- 3. Objective 3 -- OLAP Cube -----------------------------------------
    _banner("PHASE 3 -- Objective 3: OLAP Cube")
    t3 = time.time()
    olap_cube = run_olap(df_segmented, OUTPUT_DIR)
    _milestone("Objective 3 (OLAP)", t3)

    # -- Summary --------------------------------------------------------------
    _banner("PIPELINE COMPLETE")
    total = time.time() - pipeline_start
    print("  Total execution time : {:.2f}s".format(total))
    print("  Output directory     : {}".format(OUTPUT_DIR))
    print("  Generated artefacts  :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print("    - {:40s} ({:7.1f} KB)".format(f, size_kb))
    print()


if __name__ == "__main__":
    main()
