"""
Source - Data Warehouse & ETL Automation Setup
==================================================
Creates the SQLite Relational Database, defines the Star Schema, 
and executes the ETL (Extract, Transform, Load) pipeline from the raw CSV.

File Location: source/database_setup.py
"""

import os
import sqlite3
import pandas as pd


def build_star_schema(csv_path, db_path):
    """Reads raw TaiwanData.csv and populates the SQLite Star Schema database.

    Parameters
    ----------
    csv_path : str
        Path to the input raw CSV file.
    db_path : str
        Target path where the SQLite database (.db) file will be saved.
    """
    print(f"\n📡 [ETL] Initializing Data Warehouse Database at:\n     {db_path}")

    # Ensure input CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ [ERROR] Raw data file nahi mili is path par: {csv_path}"
        )

    # -------------------------------------------------------------------------
    # 1. EXTRACT: Read the Raw Flat File Data
    # -------------------------------------------------------------------------
    print("📥 [ETL] Extracting data from raw CSV file...")
    df = pd.read_csv(csv_path)

    # -------------------------------------------------------------------------
    # 2. TRANSFORM: Clean Data & Define Dimension Mapping Rules
    # -------------------------------------------------------------------------
    print("🛠️ [ETL] Transforming data & preparing dimension encodings...")

    # Data Cleaning standardisation (Jaise aapke main models me thii)
    df["EDUCATION"] = df["EDUCATION"].clip(1, 4)
    df["MARRIAGE"] = df["MARRIAGE"].clip(1, 3)

    # Human-readable mapping definitions for OLAP / Dimensional analysis
    gender_map = {1: "Male", 2: "Female"}
    edu_map = {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others",
    }
    marriage_map = {1: "Married", 2: "Single", 3: "Others"}

    def get_age_group(age):
        if age < 25:
            return "Under 25"
        elif age <= 35:
            return "25-35"
        elif age <= 45:
            return "36-45"
        elif age <= 55:
            return "46-55"
        else:
            return "Above 55"

    # -------------------------------------------------------------------------
    # 3. LOAD: Connect to DB and Create Star Schema Tables (DDL)
    # -------------------------------------------------------------------------
    # Ensure target directory exists before establishing connection
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enforce Foreign Key relational constraints in SQLite backend
    cursor.execute("PRAGMA foreign_keys = ON;")

    print("🏗️ [ETL] Building Star Schema structural layout tables...")

    # DIMENSION TABLE 1: Customer Profile Details
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Customer (
        Customer_ID INTEGER PRIMARY KEY,
        Gender TEXT,
        Education_Level TEXT,
        Marital_Status TEXT
    )
    """)

    # DIMENSION TABLE 2: Demographics & Time-less Attributes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Demographics (
        Demographic_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Age INTEGER,
        Age_Group TEXT
    )
    """)

    # CENTRAL FACT TABLE: All transaction bills, metrics and target tags
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Fact_Credit_Risk (
        Fact_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Customer_ID INTEGER,
        Demographic_ID INTEGER,
        LIMIT_BAL REAL,
        PAY_0 INTEGER, PAY_2 INTEGER, PAY_3 INTEGER, 
        PAY_4 INTEGER, PAY_5 INTEGER, PAY_6 INTEGER,
        BILL_AMT1 REAL, BILL_AMT2 REAL, BILL_AMT3 REAL, 
        BILL_AMT4 REAL, BILL_AMT5 REAL, BILL_AMT6 REAL,
        PAY_AMT1 REAL, PAY_AMT2 REAL, PAY_AMT3 REAL, 
        PAY_AMT4 REAL, PAY_AMT5 REAL, PAY_AMT6 REAL,
        Default_Status INTEGER,
        FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID),
        FOREIGN KEY (Demographic_ID) REFERENCES Dim_Demographics(Demographic_ID)
    )
    """)

    # -------------------------------------------------------------------------
    # 4. LOAD CONTINUED: Insert Data Row-by-Row inside tables
    # -------------------------------------------------------------------------
    print("🚀 [ETL] Loading structured mappings into Relational Tables...")

    # For faster throughput loading performance, we use a single transaction block
    try:
        for idx, row in df.iterrows():
            cust_id = int(row["ID"])
            age = int(row["AGE"])

            # A. Populate Dim_Demographics & capture the identity sequence ID
            cursor.execute(
                "INSERT INTO Dim_Demographics (Age, Age_Group) VALUES (?, ?)",
                (age, get_age_group(age)),
            )
            demo_id = cursor.lastrowid

            # B. Populate Dim_Customer (IGNORE if duplicate customer keys exist)
            cursor.execute(
                """
                INSERT OR IGNORE INTO Dim_Customer (Customer_ID, Gender, Education_Level, Marital_Status)
                VALUES (?, ?, ?, ?)
            """,
                (
                    cust_id,
                    gender_map.get(row["SEX"], "Others"),
                    edu_map.get(row["EDUCATION"], "Others"),
                    marriage_map.get(row["MARRIAGE"], "Others"),
                ),
            )

            # C. Populate Central Fact Table linking foreign entities
            cursor.execute(
                """
                INSERT INTO Fact_Credit_Risk (
                    Customer_ID, Demographic_ID, LIMIT_BAL, 
                    PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
                    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
                    Default_Status
                ) VALUES (?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?)
            """,
                (
                    cust_id,
                    demo_id,
                    row["LIMIT_BAL"],
                    row["PAY_0"],
                    row["PAY_2"],
                    row["PAY_3"],
                    row["PAY_4"],
                    row["PAY_5"],
                    row["PAY_6"],
                    row["BILL_AMT1"],
                    row["BILL_AMT2"],
                    row["BILL_AMT3"],
                    row["BILL_AMT4"],
                    row["BILL_AMT5"],
                    row["BILL_AMT6"],
                    row["PAY_AMT1"],
                    row["PAY_AMT2"],
                    row["PAY_AMT3"],
                    row["PAY_AMT4"],
                    row["PAY_AMT5"],
                    row["PAY_AMT6"],
                    int(row["class"]),
                ),
            )

        # Apply changes to disk file
        conn.commit()
        print(
            "✅ [ETL] Star Schema built and data loaded perfectly without issues."
        )

    except Exception as e:
        conn.rollback()
        print(f"❌ [ETL ERROR] Transaction failed! Rollback initiated: {e}")
        raise e

    finally:
        conn.close()


if __name__ == "__main__":
    # Test execution for debugging paths locally
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_CSV = os.path.join(SCRIPT_DIR, "../data/TaiwanData.csv")
    TEST_DB = os.path.join(SCRIPT_DIR, "../data/Warehouse_Credit_Risk.db")

    print("🤖 Running local isolated module validation test...")
    build_star_schema(TEST_CSV, TEST_DB)