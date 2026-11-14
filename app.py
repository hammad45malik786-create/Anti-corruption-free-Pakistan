"""
Corruption Free Pakistan — AI Public Spending Risk Radar
-------------------------------------------------------
Hackathon prototype using rule-based procurement red flags, scikit-learn
Isolation Forest anomaly detection, Benford's Law first-digit analysis,
and optional Gemini AI bilingual (English + Urdu) risk summarization.

DISCLAIMER: Outputs indicate statistical/rule-based risk only and do NOT
constitute proof of corruption. All flagged records require human audit.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Optional Gemini SDK (google-genai)
try:
    from google import genai

    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


# ==============================================================================
# PAGE CONFIG & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Corruption Free Pakistan | AI Risk Radar",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .demo-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
        margin-bottom: 0.4rem;
    }
    .real-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #10B981;
        margin-bottom: 0.4rem;
    }
    .disclaimer-box {
        padding: 0.85rem 1.1rem;
        border-radius: 0.5rem;
        background-color: #FFF7ED;
        border-left: 5px solid #EA580C;
        color: #7C2D12;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CONSTANTS & GEOGRAPHIC METADATA
# ==============================================================================
REGIONS: List[str] = [
    "Punjab",
    "Sindh",
    "Khyber Pakhtunkhwa",
    "Balochistan",
    "Islamabad Capital Territory (Federal)",
    "Gilgit-Baltistan",
    "Azad Jammu and Kashmir",
]

DEPARTMENTS: List[str] = [
    "Health",
    "Education",
    "Roads and Infrastructure",
    "Water",
    "Energy",
    "Social Protection",
]

FISCAL_YEARS: List[str] = [
    "FY 2022-23",
    "FY 2023-24",
    "FY 2024-25",
    "FY 2025-26",
]

REQUIRED_COLUMNS: List[str] = [
    "project_id",
    "region",
    "district",
    "department",
    "fiscal_year",
    "allocated_amount_pkr",
    "released_amount_pkr",
    "spent_amount_pkr",
    "contractor_id",
    "bidders_count",
    "bid_type",
    "planned_duration_days",
    "actual_duration_days",
    "completion_percent",
    "audit_objections_count",
    "date_released",
]

REGION_DISTRICTS: Dict[str, List[str]] = {
    "Punjab": [
        "Lahore",
        "Rawalpindi",
        "Multan",
        "Faisalabad",
        "Bahawalpur",
        "Gujranwala",
        "Sialkot",
        "Dera Ghazi Khan",
    ],
    "Sindh": [
        "Karachi",
        "Hyderabad",
        "Sukkur",
        "Larkana",
        "Mirpurkhas",
        "Shaheed Benazirabad",
        "Thatta",
    ],
    "Khyber Pakhtunkhwa": [
        "Peshawar",
        "Mardan",
        "Abbottabad",
        "Swat",
        "Dera Ismail Khan",
        "Kohat",
        "Mansehra",
    ],
    "Balochistan": [
        "Quetta",
        "Gwadar",
        "Turbat",
        "Khuzdar",
        "Sibi",
        "Zhob",
        "Loralai",
    ],
    "Islamabad Capital Territory (Federal)": ["Islamabad"],
    "Gilgit-Baltistan": ["Gilgit", "Skardu", "Hunza", "Diamer", "Ghizer"],
    "Azad Jammu and Kashmir": [
        "Muzaffarabad",
        "Mirpur",
        "Rawalakot",
        "Kotli",
        "Bhimber",
    ],
}

# Approximate coordinates for interactive Pakistan mapping (no external shapefile needed)
DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "Lahore": (31.5204, 74.3587),
    "Rawalpindi": (33.5651, 73.0169),
    "Multan": (30.1575, 71.5249),
    "Faisalabad": (31.4504, 73.1350),
    "Bahawalpur": (29.3544, 71.6911),
    "Gujranwala": (32.1877, 74.1945),
    "Sialkot": (32.4945, 74.5229),
    "Dera Ghazi Khan": (30.0489, 70.6455),
    "Karachi": (24.8607, 67.0011),
    "Hyderabad": (25.3960, 68.3578),
    "Sukkur": (27.7131, 68.8492),
    "Larkana": (27.5570, 68.2028),
    "Mirpurkhas": (25.5067, 69.0136),
    "Shaheed Benazirabad": (26.2442, 68.4100),
    "Thatta": (24.7475, 67.9235),
    "Peshawar": (34.0151, 71.5249),
    "Mardan": (34.1986, 72.0404),
    "Abbottabad": (34.1688, 73.2215),
    "Swat": (35.2227, 72.4258),
    "Dera Ismail Khan": (31.8626, 70.9019),
    "Kohat": (33.5889, 71.4429),
    "Mansehra": (34.3313, 73.1980),
    "Quetta": (30.1798, 66.9750),
    "Gwadar": (25.1216, 62.3254),
    "Turbat": (26.0023, 63.0440),
    "Khuzdar": (27.8165, 66.6057),
    "Sibi": (29.5430, 67.8773),
    "Zhob": (31.3417, 69.4486),
    "Loralai": (30.3705, 68.5979),
    "Islamabad": (33.6844, 73.0479),
    "Gilgit": (35.9208, 74.3144),
    "Skardu": (35.2971, 75.6333),
    "Hunza": (36.3167, 74.6500),
    "Diamer": (35.4212, 74.1003),
    "Ghizer": (36.1786, 73.7606),
    "Muzaffarabad": (34.3700, 73.4711),
    "Mirpur": (33.1480, 73.7518),
    "Rawalakot": (33.8578, 73.7604),
    "Kotli": (33.5184, 73.9022),
    "Bhimber": (32.9746, 74.0785),
}

REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Punjab": (31.1704, 72.7097),
    "Sindh": (25.8943, 68.5247),
    "Khyber Pakhtunkhwa": (34.0151, 71.5249),
    "Balochistan": (28.4907, 65.0958),
    "Islamabad Capital Territory (Federal)": (33.6844, 73.0479),
    "Gilgit-Baltistan": (35.8026, 74.9832),
    "Azad Jammu and Kashmir": (33.8578, 73.7604),
}

RISK_COLORS: Dict[str, str] = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#F97316",
    "Critical": "#DC2626",
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def format_pkr(amount: float) -> str:
    """Format PKR values into human-readable Billions/Millions."""
    if pd.isna(amount):
        return "PKR 0"
    abs_val = abs(amount)
    if abs_val >= 1e9:
        return f"PKR {amount / 1e9:,.2f} B"
    if abs_val >= 1e6:
        return f"PKR {amount / 1e6:,.2f} M"
    return f"PKR {amount:,.0f}"


def get_gemini_api_key(manual_key: str = "") -> str:
    """Safely retrieve Gemini API key from manual input or Streamlit secrets."""
    if manual_key and manual_key.strip():
        return manual_key.strip()
    try:
        for secret_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if secret_name in st.secrets and st.secrets[secret_name]:
                return str(st.secrets[secret_name]).strip()
    except Exception:
        pass
    return ""


# ==============================================================================
# SYNTHETIC DEMO DATA GENERATOR (FIXED SEED = 42)
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 42, n_rows: int = 700) -> pd.DataFrame:
    """
    Generate realistic synthetic public-sector development data across Pakistan.
    Clean records follow log-uniform distributions (naturally conforming to
    Benford's Law), while ~9% of rows contain deliberate corruption-like patterns.
    """
    rng = np.random.default_rng(seed)

    region_weights = [0.26, 0.20, 0.16, 0.14, 0.10, 0.07, 0.07]
    dept_budget_bounds = {
        "Roads and Infrastructure": (7.0, 8.6),  # PKR 10M - 400M
        "Energy": (6.9, 8.5),
        "Water": (6.6, 8.1),
        "Health": (6.5, 8.0),
        "Education": (6.3, 7.8),
        "Social Protection": (6.3, 7.7),
    }

    fy_start_dates = {
        "FY 2022-23": pd.Timestamp("2022-07-15"),
        "FY 2023-24": pd.Timestamp("2023-07-15"),
        "FY 2024-25": pd.Timestamp("2024-07-15"),
        "FY 2025-26": pd.Timestamp("2025-07-15"),
    }

    contractors = [f"CTR-PK-{i:03d}" for i in range(1, 86)]
    rows = []

    for idx in range(n_rows):
        region = str(rng.choice(REGIONS, p=region_weights))
        district = str(rng.choice(REGION_DISTRICTS[region]))
        department = str(rng.choice(DEPARTMENTS))
        fiscal_year = str(rng.choice(FISCAL_YEARS))

        # Log-uniform allocation preserves Benford's Law for legitimate projects
        low_exp, high_exp = dept_budget_bounds[department]
        raw_alloc = 10 ** rng.uniform(low_exp, high_exp)
        # Keep non-round realistic figures (avoid accidental exact 5M multiples)
        allocated = float(round(raw_alloc / 1000.0) * 1000 + rng.integers(150, 850))

        release_ratio = float(rng.uniform(0.65, 0.98))
        released = float(round(allocated * release_ratio + rng.integers(100, 900), 0))

        spend_ratio_of_rel = float(rng.uniform(0.75, 0.99))
        spent = float(round(released * spend_ratio_of_rel + rng.integers(100, 900), 0))

        # Ensure clean projects mostly have realistic completion aligned with spending
        completion = float(
            np.clip(
                (spent / allocated) * 100.0 + rng.normal(0, 6.0),
                12.0,
                100.0,
            )
        )

        bid_type = str(rng.choice(["open", "limited", "single source"], p=[0.74, 0.22, 0.04]))
        if bid_type == "open":
            bidders_count = int(rng.integers(3, 11))
        elif bid_type == "limited":
            bidders_count = int(rng.integers(2, 5))
        else:
            bidders_count = 1

        planned_days = int(rng.integers(120, 720))
        actual_days = int(max(60, planned_days * rng.uniform(0.90, 1.25)))

        audit_objections = int(rng.choice([0, 1, 2], p=[0.72, 0.22, 0.06]))
        contractor_id = str(rng.choice(contractors))

        start_dt = fy_start_dates[fiscal_year]
        date_released = (start_dt + pd.Timedelta(days=int(rng.integers(0, 320)))).strftime("%Y-%m-%d")

        rows.append(
            {
                "project_id": f"PK-{fiscal_year[-5:-3]}{fiscal_year[-2:]}-{idx + 1001:04d}",
                "region": region,
                "district": district,
                "department": department,
                "fiscal_year": fiscal_year,
                "allocated_amount_pkr": allocated,
                "released_amount_pkr": released,
                "spent_amount_pkr": spent,
                "contractor_id": contractor_id,
                "bidders_count": bidders_count,
                "bid_type": bid_type,
                "planned_duration_days": planned_days,
                "actual_duration_days": actual_days,
                "completion_percent": round(completion, 1),
                "audit_objections_count": audit_objections,
                "date_released": date_released,
            }
        )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------------------------
    # INJECT ~9% CORRUPTION-LIKE PATTERNS (63 out of 700 rows)
    # --------------------------------------------------------------------------
    anomaly_indices = rng.choice(df.index, size=63, replace=False)

    # Dedicated high-concentration cartel contractors for specific districts
    cartel_map = {
        "Quetta": "CTR-PK-091",
        "Karachi": "CTR-PK-092",
        "Lahore": "CTR-PK-093",
        "Peshawar": "CTR-PK-094",
        "Gilgit": "CTR-PK-095",
        "Muzaffarabad": "CTR-PK-096",
        "Islamabad": "CTR-PK-097",
    }

    for i, row_idx in enumerate(anomaly_indices):
        pattern = i % 5
        alloc = float(df.loc[row_idx, "allocated_amount_pkr"])
        planned = int(df.loc[row_idx, "planned_duration_days"])

        if pattern == 0:
            # Pattern 1: Ghost / stalled project — massive release & spend, tiny completion
            df.loc[row_idx, "released_amount_pkr"] = round(alloc * rng.uniform(0.92, 1.00), 0)
            df.loc[row_idx, "spent_amount_pkr"] = round(alloc * rng.uniform(0.90, 1.18), 0)
            df.loc[row_idx, "completion_percent"] = round(float(rng.uniform(5.0, 24.0)), 1)
            df.loc[row_idx, "actual_duration_days"] = int(planned * rng.uniform(1.6, 2.5))
            df.loc[row_idx, "audit_objections_count"] = int(rng.integers(4, 9))

        elif pattern == 1:
            # Pattern 2: Single-source / single-bidder contract with severe cost overrun
            df.loc[row_idx, "bid_type"] = "single source"
            df.loc[row_idx, "bidders_count"] = 1
            df.loc[row_idx, "released_amount_pkr"] = round(alloc * rng.uniform(0.98, 1.15), 0)
            df.loc[row_idx, "spent_amount_pkr"] = round(alloc * rng.uniform(1.18, 1.48), 0)
            df.loc[row_idx, "actual_duration_days"] = int(planned * rng.uniform(1.7, 2.6))
            df.loc[row_idx, "audit_objections_count"] = int(rng.integers(3, 8))

        elif pattern == 2:
            # Pattern 3: Just-below-threshold splitting OR exact suspicious round millions
            if i % 2 == 0:
                threshold = float(rng.choice([5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]))
                just_below = round(threshold * rng.uniform(0.975, 0.996), -3)
                df.loc[row_idx, "allocated_amount_pkr"] = just_below
                df.loc[row_idx, "released_amount_pkr"] = just_below
                df.loc[row_idx, "spent_amount_pkr"] = round(just_below * rng.uniform(0.99, 1.08), -3)
            else:
                round_amt = float(rng.choice([40_000_000, 50_000_000, 75_000_000, 80_000_000, 90_000_000]))
                df.loc[row_idx, "allocated_amount_pkr"] = round_amt
                df.loc[row_idx, "released_amount_pkr"] = round_amt
                df.loc[row_idx, "spent_amount_pkr"] = round_amt
            df.loc[row_idx, "bid_type"] = "limited"
            df.loc[row_idx, "bidders_count"] = int(rng.choice([1, 2]))
            df.loc[row_idx, "audit_objections_count"] = int(rng.integers(2, 6))

        elif pattern == 3:
            # Pattern 4: District contractor monopoly / repeat offender
            target_district = str(rng.choice(list(cartel_map.keys())))
            # Find matching region for that district
            target_region = next(r for r, ds in REGION_DISTRICTS.items() if target_district in ds)
            df.loc[row_idx, "region"] = target_region
            df.loc[row_idx, "district"] = target_district
            df.loc[row_idx, "contractor_id"] = cartel_map[target_district]
            df.loc[row_idx, "allocated_amount_pkr"] = round(float(rng.uniform(90_000_000, 260_000_000)), 0)
            df.loc[row_idx, "released_amount_pkr"] = round(df.loc[row_idx, "allocated_amount_pkr"] * 0.98, 0)
            df.loc[row_idx, "spent_amount_pkr"] = round(df.loc[row_idx, "allocated_amount_pkr"] * rng.uniform(1.06, 1.25), 0)
            df.loc[row_idx, "bidders_count"] = 1
            df.loc[row_idx, "bid_type"] = "single source"
            df.loc[row_idx, "actual_duration_days"] = int(planned * rng.uniform(1.6, 2.3))
            df.loc[row_idx, "audit_objections_count"] = int(rng.integers(3, 7))

        else:
            # Pattern 5: Chronic delays + heavy audit objections + budget overrun
            df.loc[row_idx, "actual_duration_days"] = int(planned * rng.uniform(2.0, 3.1))
            df.loc[row_idx, "spent_amount_pkr"] = round(alloc * rng.uniform(1.12, 1.38), 0)
            df.loc[row_idx, "released_amount_pkr"] = round(alloc * rng.uniform(0.95, 1.05), 0)
            df.loc[row_idx, "completion_percent"] = round(float(rng.uniform(25.0, 55.0)), 1)
            df.loc[row_idx, "audit_objections_count"] = int(rng.integers(5, 11))

    return df


# ==============================================================================
# CSV VALIDATION & PREPROCESSING
# ==============================================================================
def validate_and_clean_csv(uploaded_df: pd.DataFrame) -> Tuple[pd.DataFrame | None, str]:
    """Validate uploaded CSV columns and safely coerce types."""
    df = uploaded_df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Uploaded CSV is missing required columns: {', '.join(missing)}"

    numeric_cols = [
        "allocated_amount_pkr",
        "released_amount_pkr",
        "spent_amount_pkr",
        "bidders_count",
        "planned_duration_days",
        "actual_duration_days",
        "completion_percent",
        "audit_objections_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["allocated_amount_pkr"] = df["allocated_amount_pkr"].clip(lower=1.0)
    df["planned_duration_days"] = df["planned_duration_days"].clip(lower=1)
    df["completion_percent"] = df["completion_percent"].clip(lower=0.0, upper=100.0)
    df["bid_type"] = df["bid_type"].astype(str).str.strip().str.lower()

    for text_col in ["project_id", "region", "district", "department", "fiscal_year", "contractor_id", "date_released"]:
        df[text_col] = df[text_col].astype(str).str.strip()

    return df, ""


# ==============================================================================
# AI & RULE-BASED RISK ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_risk_engine(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine domain-specific procurement red-flag rules and scikit-learn
    Isolation Forest anomaly detection to score every project from 0 to 100
    and generate plain-English explanations.
    """
    df = raw_df.copy()

    # 1. Engineered Features
    df["spend_to_allocation_ratio"] = df["spent_amount_pkr"] / df["allocated_amount_pkr"].clip(lower=1.0)
    df["release_to_allocation_ratio"] = df["released_amount_pkr"] / df["allocated_amount_pkr"].clip(lower=1.0)
    df["cost_per_completion_percent"] = df["spent_amount_pkr"] / df["completion_percent"].clip(lower=1.0)
    df["log_cost_per_completion"] = np.log1p(df["cost_per_completion_percent"])
    df["spend_completion_gap"] = (df["spend_to_allocation_ratio"] * 100.0) - df["completion_percent"]
    df["delay_ratio"] = df["actual_duration_days"] / df["planned_duration_days"].clip(lower=1.0)

    # Contractor concentration within each (region, district)
    dist_totals = (
        df.groupby(["region", "district"])
        .agg(
            district_total_pkr=("allocated_amount_pkr", "sum"),
            district_project_count=("project_id", "count"),
        )
        .reset_index()
    )
    ctr_dist = (
        df.groupby(["region", "district", "contractor_id"])
        .agg(
            ctr_dist_pkr=("allocated_amount_pkr", "sum"),
            ctr_dist_projects=("project_id", "count"),
        )
        .reset_index()
    )
    df = df.merge(dist_totals, on=["region", "district"], how="left")
    df = df.merge(ctr_dist, on=["region", "district", "contractor_id"], how="left")
    df["contractor_concentration"] = df["ctr_dist_pkr"] / df["district_total_pkr"].clip(lower=1.0)

    # Contractor-level repeat overrun & delay history
    df["_is_overrun_delay"] = (df["delay_ratio"] >= 1.35) & (df["spend_to_allocation_ratio"] >= 1.05)
    ctr_repeat = (
        df.groupby("contractor_id")["_is_overrun_delay"]
        .sum()
        .rename("contractor_repeat_overrun_delays")
        .reset_index()
    )
    df = df.merge(ctr_repeat, on="contractor_id", how="left")

    # 2. Isolation Forest Anomaly Detection
    iso_features = [
        "spend_to_allocation_ratio",
        "log_cost_per_completion",
        "spend_completion_gap",
        "delay_ratio",
        "contractor_concentration",
        "bidders_count",
        "audit_objections_count",
    ]
    X = df[iso_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
    )
    iso_preds = iso_model.fit_predict(X_scaled)
    # decision_function: higher = more normal, lower = more anomalous
    raw_dec = iso_model.decision_function(X_scaled)
    inv_dec = -raw_dec
    min_d, max_d = float(inv_dec.min()), float(inv_dec.max())
    if max_d > min_d:
        df["iso_anomaly_score"] = ((inv_dec - min_d) / (max_d - min_d) * 100.0).round(1)
    else:
        df["iso_anomaly_score"] = 0.0
    df["iso_is_anomaly"] = iso_preds == -1

    # 3. Rule-Based Red Flags & Plain-English Explanations
    approval_thresholds = [5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]

    rule_scores: List[float] = []
    explanations: List[str] = []
    flag_tags_list: List[List[str]] = []

    for row in df.itertuples(index=False):
        pts = 0.0
        reasons: List[str] = []
        tags: List[str] = []

        # Rule A: Spending above allocation / release
        if row.spent_amount_pkr > row.allocated_amount_pkr * 1.02:
            over_pct = (row.spend_to_allocation_ratio - 1.0) * 100.0
            pts += 26.0
            tags.append("Spending Above Allocation")
            reasons.append(
                f"Spent ({format_pkr(row.spent_amount_pkr)}) exceeds allocated budget "
                f"({format_pkr(row.allocated_amount_pkr)}) by {over_pct:.1f}%"
            )
        elif row.spent_amount_pkr > row.released_amount_pkr * 1.02:
            pts += 16.0
            tags.append("Spending Above Release")
            reasons.append(
                f"Spent ({format_pkr(row.spent_amount_pkr)}) exceeds released funds "
                f"({format_pkr(row.released_amount_pkr)})"
            )

        # Rule B: Large release with low completion
        if row.release_to_allocation_ratio >= 0.75 and row.completion_percent < 40.0:
            pts += 26.0
            tags.append("High Release / Low Completion")
            reasons.append(
                f"{row.release_to_allocation_ratio * 100:.0f}% of budget released while "
                f"physical completion is only {row.completion_percent:.1f}%"
            )

        # Rule C: Single-bidder or single-source contract
        bid_type_clean = str(row.bid_type).lower()
        if bid_type_clean == "single source" or row.bidders_count <= 1:
            pts += 22.0
            tags.append("Single-Source / Single-Bidder")
            reasons.append(
                f"Non-competitive award (bid type: '{row.bid_type}', bidders: {int(row.bidders_count)})"
            )

        # Rule D: District contractor concentration
        if (
            row.contractor_concentration >= 0.35
            and row.ctr_dist_projects >= 2
            and row.district_project_count >= 3
        ):
            pts += 18.0
            tags.append("District Contractor Monopoly")
            reasons.append(
                f"Contractor {row.contractor_id} holds {row.contractor_concentration * 100:.1f}% "
                f"of {row.district} contract value ({int(row.ctr_dist_projects)} projects)"
            )

        # Rule E: Repeated cost overruns & severe delays
        delay_triggered = row.delay_ratio >= 1.50
        repeat_triggered = row.contractor_repeat_overrun_delays >= 2 and row._is_overrun_delay
        if delay_triggered or repeat_triggered:
            if delay_triggered and repeat_triggered:
                pts += 22.0
                tags.append("Repeated Overruns & Delays")
                reasons.append(
                    f"Severe schedule delay ({row.delay_ratio:.2f}x planned duration) by repeat-overrun "
                    f"contractor ({int(row.contractor_repeat_overrun_delays)} troubled contracts)"
                )
            elif delay_triggered:
                pts += 14.0
                tags.append("Severe Delay")
                reasons.append(
                    f"Schedule overrun: {int(row.actual_duration_days)} days vs "
                    f"{int(row.planned_duration_days)} planned ({row.delay_ratio:.2f}x)"
                )
            else:
                pts += 14.0
                tags.append("Repeated Contractor Overruns")
                reasons.append(
                    f"Contractor {row.contractor_id} has {int(row.contractor_repeat_overrun_delays)} "
                    f"projects with simultaneous cost overruns and delays"
                )

        # Rule F: Many audit objections
        if row.audit_objections_count >= 6:
            pts += 22.0
            tags.append("High Audit Objections")
            reasons.append(f"{int(row.audit_objections_count)} AGP/internal audit objections logged")
        elif row.audit_objections_count >= 3:
            pts += 15.0
            tags.append("High Audit Objections")
            reasons.append(f"{int(row.audit_objections_count)} audit objections logged")

        # Rule G: Round-number or just-below-threshold amounts
        just_below_hit = False
        round_hit = False
        for amt in (row.allocated_amount_pkr, row.spent_amount_pkr):
            for t in approval_thresholds:
                if 0.96 * t <= amt < t:
                    just_below_hit = True
                    break
            if amt >= 5_000_000 and abs(amt % 5_000_000) < 1.0:
                round_hit = True

        if just_below_hit:
            pts += 15.0
            tags.append("Just-Below-Threshold Amount")
            reasons.append(
                f"Contract value ({format_pkr(row.allocated_amount_pkr)}) sits just below a "
                f"standard procurement/PC-I approval threshold"
            )
        elif round_hit:
            pts += 12.0
            tags.append("Suspicious Round-Number Amount")
            reasons.append(
                f"Exact round multi-million amount ({format_pkr(row.allocated_amount_pkr)} / "
                f"{format_pkr(row.spent_amount_pkr)})"
            )

        # Statistical outlier note from Isolation Forest
        if row.iso_is_anomaly:
            tags.append("AI Isolation Forest Outlier")
            reasons.append(
                f"Flagged as multi-feature statistical outlier by Isolation Forest "
                f"(AI anomaly score: {row.iso_anomaly_score:.0f}/100)"
            )

        rule_score = min(100.0, pts)
        rule_scores.append(rule_score)
        flag_tags_list.append(tags)

        if reasons:
            explanations.append("; ".join(reasons) + ".")
        else:
            explanations.append("No major rule red flags detected; spending and delivery profile within normal bounds.")

    df["rule_score"] = np.round(rule_scores, 1)
    df["flag_tags"] = flag_tags_list
    df["flags_count"] = df["flag_tags"].apply(len)
    df["risk_explanation"] = explanations

    # 4. Combine Rule Score + Isolation Forest Score into Final Risk Score (0-100)
    combined = 0.68 * df["rule_score"] + 0.32 * df["iso_anomaly_score"]
    # Ensure severe rule violations are not overly diluted
    combined = np.where(df["rule_score"] >= 65, np.maximum(combined, 0.85 * df["rule_score"]), combined)
    df["risk_score"] = np.clip(combined, 0.0, 100.0).round(1)

    def categorize_risk(score: float) -> str:
        if score >= 75.0:
            return "Critical"
        if score >= 50.0:
            return "High"
        if score >= 25.0:
            return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(categorize_risk)
    df["is_flagged"] = df["risk_level"].isin(["High", "Critical"])

    # Attach coordinates for mapping
    df["lat"] = df.apply(
        lambda r: DISTRICT_COORDS.get(r["district"], REGION_CENTROIDS.get(r["region"], (30.3753, 69.3451)))[0],
        axis=1,
    )
    df["lon"] = df.apply(
        lambda r: DISTRICT_COORDS.get(r["district"], REGION_CENTROIDS.get(r["region"], (30.3753, 69.3451)))[1],
        axis=1,
    )

    df.drop(columns=["_is_overrun_delay"], inplace=True)
    return df


# ==============================================================================
# BENFORD'S LAW ANALYSIS
# ==============================================================================
def compute_benford_stats(series: pd.Series) -> Tuple[pd.DataFrame, float, float, float, str, int]:
    """Compute first-digit frequencies vs Benford's Law expected distribution."""
    digits = np.arange(1, 10)
    expected_probs = np.array([math.log10(1.0 + 1.0 / d) for d in digits])

    clean_vals = pd.to_numeric(series, errors="coerce").dropna()
    clean_vals = clean_vals[clean_vals > 0]

    leading_digits = (
        clean_vals.astype(str)
        .str.lstrip("0.")
        .str[0]
    )
    leading_digits = pd.to_numeric(leading_digits, errors="coerce").dropna().astype(int)
    leading_digits = leading_digits[leading_digits.between(1, 9)]

    n = int(len(leading_digits))
    counts = leading_digits.value_counts().reindex(digits, fill_value=0).values.astype(float)

    if n == 0:
        actual_probs = np.zeros(9)
        mad = 0.0
        chi2_stat, p_val = 0.0, 1.0
        conformity = "Insufficient Data"
    else:
        actual_probs = counts / n
        mad = float(np.mean(np.abs(actual_probs - expected_probs)))
        expected_counts = expected_probs * n
        chi2_stat, p_val = chisquare(f_obs=counts, f_exp=expected_counts)
        chi2_stat, p_val = float(chi2_stat), float(p_val)

        if n < 30:
            conformity = "Low Sample Size (<30)"
        elif mad < 0.006:
            conformity = "Close Conformity"
        elif mad < 0.012:
            conformity = "Acceptable Conformity"
        elif mad < 0.015:
            conformity = "Marginally Acceptable"
        else:
            conformity = "Nonconformity (Investigate)"

    benford_df = pd.DataFrame(
        {
            "Digit": digits,
            "Actual (%)": np.round(actual_probs * 100.0, 2),
            "Expected Benford (%)": np.round(expected_probs * 100.0, 2),
            "Count": counts.astype(int),
        }
    )
    return benford_df, mad, chi2_stat, p_val, conformity, n


# ==============================================================================
# GEMINI AI BILINGUAL ASSISTANT
# ==============================================================================
def generate_bilingual_summary(project_row: pd.Series, api_key: str) -> str:
    """Use google-genai SDK to explain a project's risk in simple English & Urdu."""
    if not GEMINI_SDK_AVAILABLE:
        return "Error: `google-genai` package is not installed."

    prompt = f"""
You are an anti-corruption data analyst helping Pakistani citizens and government auditors.
Summarize the procurement and spending risk indicators for the following public project.

IMPORTANT RULES:
- Clearly state that these are risk indicators requiring human audit, NOT proof of corruption.
- Keep the tone objective, neutral, and easy to understand for a non-technical reader.
- Provide two clearly labeled sections:
  1) **English Summary** (3–4 short bullet points + recommended next audit step)
  2) **Urdu Summary (اردو خلاصہ)** (simple, natural Urdu explaining the same points clearly)

PROJECT RECORD:
- Project ID: {project_row['project_id']}
- Region / District: {project_row['region']} / {project_row['district']}
- Department: {project_row['department']} ({project_row['fiscal_year']})
- Allocated Budget: {format_pkr(project_row['allocated_amount_pkr'])}
- Released Amount: {format_pkr(project_row['released_amount_pkr'])}
- Spent Amount: {format_pkr(project_row['spent_amount_pkr'])}
- Physical Completion: {project_row['completion_percent']}%
- Contractor ID: {project_row['contractor_id']}
- Bid Type / Bidders: {project_row['bid_type']} ({int(project_row['bidders_count'])} bidder(s))
- Planned vs Actual Duration: {int(project_row['planned_duration_days'])} vs {int(project_row['actual_duration_days'])} days
- Audit Objections: {int(project_row['audit_objections_count'])}
- AI Risk Score: {project_row['risk_score']}/100 ({project_row['risk_level']})
- Detected Risk Factors: {project_row['risk_explanation']}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text or "No response returned by Gemini."


# ==============================================================================
# SIDEBAR: DATA SOURCE & GLOBAL FILTERS
# ==============================================================================
with st.sidebar:
    st.title("🇵🇰 Control Panel")
    st.caption("Configure data source & global dashboard filters.")

    st.subheader("1. Data Source")
    uploaded_file = st.file_uploader(
        "Upload Real Project CSV (Optional)",
        type=["csv"],
        help="Replace synthetic demo data with real procurement/audit CSVs.",
    )

    demo_raw = generate_demo_data(seed=42, n_rows=700)

    if uploaded_file is not None:
        try:
            user_raw = pd.read_csv(uploaded_file)
            cleaned_df, err_msg = validate_and_clean_csv(user_raw)
            if cleaned_df is None:
                st.error(err_msg)
                st.info("Reverting to built-in DEMO DATA.")
                active_raw_df = demo_raw
                is_demo = True
            else:
                active_raw_df = cleaned_df
                is_demo = False
                st.success(f"Loaded {len(active_raw_df):,} rows from uploaded CSV.")
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            active_raw_df = demo_raw
            is_demo = True
    else:
        active_raw_df = demo_raw
        is_demo = True

    with st.expander("Real Data Sources & CSV Template"):
        st.markdown(
            """
            In production, this dashboard integrates directly with:
            - **PPRA** (Public Procurement Regulatory Authority) tender & award feeds
            - **Auditor General of Pakistan (AGP)** audit para datasets
            - **Provincial ADP & Finance Dept** budget release/expenditure sheets
            """
        )
        template_csv = demo_raw.head(10).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sample CSV Template",
            data=template_csv,
            file_name="pakistan_projects_template.csv",
            mime="text/csv",
            width="stretch",
        )

    # Run Risk Engine on active dataset
    scored_df = run_risk_engine(active_raw_df)

    st.divider()
    st.subheader("2. Global Filters")

    avail_regions = [r for r in REGIONS if r in scored_df["region"].unique()] or sorted(
        scored_df["region"].unique().tolist()
    )
    selected_regions = st.multiselect(
        "Region / Province",
        options=avail_regions,
        default=avail_regions,
    )

    avail_years = sorted(scored_df["fiscal_year"].unique().tolist())
    selected_years = st.multiselect(
        "Fiscal Year",
        options=avail_years,
        default=avail_years,
    )

    avail_depts = [d for d in DEPARTMENTS if d in scored_df["department"].unique()] or sorted(
        scored_df["department"].unique().tolist()
    )
    selected_depts = st.multiselect(
        "Department / Sector",
        options=avail_depts,
        default=avail_depts,
    )

    st.divider()
    st.subheader("3. Gemini AI Key (Optional)")
    manual_gemini_key = st.text_input(
        "Temporary Gemini API Key",
        type="password",
        help="Optional override if GEMINI_API_KEY is not configured in Streamlit Secrets.",
    )


# ==============================================================================
# HEADER & ETHICAL DISCLAIMER
# ==============================================================================
col_title, col_badge = st.columns([4, 1.3])
with col_title:
    st.title("Corruption Free Pakistan")
    st.caption("AI & Statistical Risk Radar for Government Fund Allocation and Public Procurement")
with col_badge:
    st.write("")
    if is_demo:
        st.markdown('<div class="demo-badge">⚠️ DEMO DATA (Synthetic Seed=42)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="real-badge">✅ UPLOADED CSV ACTIVE</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Important Legal & Ethical Disclaimer:</strong> This tool highlights statistical anomalies and
        procurement risk indicators—it does <strong>not</strong> constitute proof of corruption or wrongdoing.
        Every flagged project requires formal human audit, physical site inspection, and due process before any conclusion.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# APPLY GLOBAL FILTERS
# ==============================================================================
filtered_df = scored_df[
    scored_df["region"].isin(selected_regions)
    & scored_df["fiscal_year"].isin(selected_years)
    & scored_df["department"].isin(selected_depts)
].copy()

if filtered_df.empty:
    st.warning("No projects match your selected sidebar filters. Please select at least one Region, Fiscal Year, and Department.")
    st.stop()


# ==============================================================================
# DASHBOARD TABS / PAGES
# ==============================================================================
tab_overview, tab_map, tab_region, tab_contractor, tab_explorer, tab_benford = st.tabs(
    [
        "📊 Overview",
        "🗺️ Pakistan Risk Map",
        "⚖️ Region Comparison",
        "🏢 Contractor Watch",
        "🔍 Project Explorer & AI Assistant",
        "📐 Benford's Law Analysis",
    ]
)


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    total_funds = float(filtered_df["allocated_amount_pkr"].sum())
    flagged_df = filtered_df[filtered_df["is_flagged"]]
    funds_at_risk = float(flagged_df["allocated_amount_pkr"].sum())
    n_flagged = len(flagged_df)
    pct_flagged = (n_flagged / len(filtered_df)) * 100.0

    region_avg = filtered_df.groupby("region")["risk_score"].mean().sort_values(ascending=False)
    highest_risk_region = str(region_avg.index[0])
    highest_risk_score = float(region_avg.iloc[0])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Allocated Funds", format_pkr(total_funds), f"{len(filtered_df):,} projects")
    k2.metric(
        "Funds in Flagged Projects",
        format_pkr(funds_at_risk),
        f"{(funds_at_risk / total_funds * 100.0):.1f}% of allocated",
        delta_color="inverse",
    )
    k3.metric(
        "Flagged Projects (High/Critical)",
        f"{n_flagged:,}",
        f"{pct_flagged:.1f}% of filtered",
        delta_color="inverse",
    )
    k4.metric(
        "Highest-Risk Region",
        highest_risk_region.replace(" (Federal)", ""),
        f"Avg Risk: {highest_risk_score:.1f} / 100",
        delta_color="inverse",
    )

    st.divider()

    c_left, c_right = st.columns(2)
    with c_left:
        level_counts = (
            filtered_df["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
            .reset_index()
        )
        level_counts.columns = ["Risk Level", "Projects"]
        fig_donut = px.pie(
            level_counts,
            names="Risk Level",
            values="Projects",
            hole=0.48,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            title="Project Distribution by AI Risk Level",
        )
        fig_donut.update_traces(textinfo="label+percent+value")
        st.plotly_chart(fig_donut, width="stretch")

    with c_right:
        fig_hist = px.histogram(
            filtered_df,
            x="risk_score",
            color="risk_level",
            nbins=25,
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": ["Low", "Medium", "High", "Critical"]},
            title="Risk Score Distribution (0–100)",
            labels={"risk_score": "Final Risk Score (0–100)", "risk_level": "Risk Level"},
        )
        fig_hist.add_vline(x=50, line_dash="dash", line_color="#DC2626", annotation_text="Flag Threshold (50)")
        st.plotly_chart(fig_hist, width="stretch")

    # Breakdown of most frequent red flags
    all_tags = [tag for tags in filtered_df["flag_tags"] for tag in tags]
    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts().reset_index()
        tag_counts.columns = ["Red Flag Trigger", "Occurrences"]
        fig_tags = px.bar(
            tag_counts.sort_values("Occurrences"),
            x="Occurrences",
            y="Red Flag Trigger",
            orientation="h",
            color="Occurrences",
            color_continuous_scale="OrRd",
            title="Most Frequent Red Flags & AI Outliers Detected",
        )
        fig_tags.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig_tags, width="stretch")

    # Quick CSV Export on Overview
    flagged_csv_bytes = flagged_df.drop(columns=["flag_tags", "lat", "lon"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Flagged Projects Report (CSV)",
        data=flagged_csv_bytes,
        file_name="corruption_free_pakistan_flagged_projects.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------------------
# TAB 2: PAKISTAN RISK MAP
# ------------------------------------------------------------------------------
with tab_map:
    st.subheader("Geographic Risk Distribution Across Pakistan")
    st.caption(
        "Interactive district-level map (bubble size = allocated funds, color = average AI risk score) "
        "paired with regional average risk rankings."
    )

    district_agg = (
        filtered_df.groupby(["region", "district", "lat", "lon"])
        .agg(
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            total_allocated=("allocated_amount_pkr", "sum"),
            flagged_count=("is_flagged", "sum"),
            project_count=("project_id", "count"),
        )
        .reset_index()
    )
    district_agg["Allocated (Formatted)"] = district_agg["total_allocated"].apply(format_pkr)

    map_col, bar_col = st.columns([1.45, 1.0])
    with map_col:
        fig_map = px.scatter_map(
            district_agg,
            lat="lat",
            lon="lon",
            size="total_allocated",
            color="avg_risk",
            hover_name="district",
            hover_data={
                "region": True,
                "avg_risk": ":.1f",
                "flagged_count": True,
                "project_count": True,
                "Allocated (Formatted)": True,
                "lat": False,
                "lon": False,
                "total_allocated": False,
            },
            color_continuous_scale=["#10B981", "#F59E0B", "#EA580C", "#DC2626"],
            range_color=(0, 80),
            size_max=28,
            zoom=4.4,
            center={"lat": 30.3753, "lon": 69.3451},
            map_style="carto-positron",
            title="Pakistan District Risk Heatmap",
            labels={"avg_risk": "Avg Risk Score"},
        )
        fig_map.update_layout(height=540, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, width="stretch")

    with bar_col:
        reg_risk = (
            filtered_df.groupby("region")
            .agg(
                avg_risk=("risk_score", "mean"),
                flagged_pct=("is_flagged", lambda s: s.mean() * 100.0),
                flagged_count=("is_flagged", "sum"),
                projects=("project_id", "count"),
            )
            .reset_index()
            .sort_values("avg_risk", ascending=True)
        )
        fig_reg_bar = px.bar(
            reg_risk,
            x="avg_risk",
            y="region",
            orientation="h",
            color="avg_risk",
            color_continuous_scale=["#10B981", "#F59E0B", "#EA580C", "#DC2626"],
            range_color=(0, 80),
            text=reg_risk["avg_risk"].round(1),
            hover_data={"flagged_count": True, "projects": True, "flagged_pct": ":.1f"},
            title="Average Risk Score by Region",
            labels={"avg_risk": "Avg Risk Score (0–100)", "region": "Region"},
        )
        fig_reg_bar.update_traces(textposition="outside")
        fig_reg_bar.update_layout(height=540, coloraxis_showscale=False)
        st.plotly_chart(fig_reg_bar, width="stretch")


# ------------------------------------------------------------------------------
# TAB 3: REGION COMPARISON
# ------------------------------------------------------------------------------
with tab_region:
    st.subheader("Regional Comparison: Budget Allocation, Spending & Risk")

    reg_comp = (
        filtered_df.groupby("region")
        .agg(
            Allocated_PKR=("allocated_amount_pkr", "sum"),
            Released_PKR=("released_amount_pkr", "sum"),
            Spent_PKR=("spent_amount_pkr", "sum"),
            Avg_Risk=("risk_score", "mean"),
            Flagged_Projects=("is_flagged", "sum"),
            Total_Projects=("project_id", "count"),
            Funds_At_Risk_PKR=("allocated_amount_pkr", lambda s: s[filtered_df.loc[s.index, "is_flagged"]].sum()),
        )
        .reset_index()
    )
    reg_comp["Flagged_Rate_%"] = (reg_comp["Flagged_Projects"] / reg_comp["Total_Projects"] * 100.0).round(1)

    # Grouped Bar: Allocated vs Released vs Spent (in Billions PKR)
    melt_df = reg_comp.melt(
        id_vars="region",
        value_vars=["Allocated_PKR", "Released_PKR", "Spent_PKR"],
        var_name="Fund Stage",
        value_name="Amount (PKR)",
    )
    melt_df["Amount (PKR Billions)"] = melt_df["Amount (PKR)"] / 1e9
    melt_df["Fund Stage"] = melt_df["Fund Stage"].str.replace("_PKR", "")

    fig_alloc_spent = px.bar(
        melt_df,
        x="region",
        y="Amount (PKR Billions)",
        color="Fund Stage",
        barmode="group",
        color_discrete_map={"Allocated": "#2563EB", "Released": "#0EA5E9", "Spent": "#F97316"},
        title="Allocated vs Released vs Spent Funds by Region (PKR Billions)",
    )
    fig_alloc_spent.update_layout(xaxis_title="", xaxis_tickangle=-20)
    st.plotly_chart(fig_alloc_spent, width="stretch")

    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        reg_comp["Allocated_B"] = reg_comp["Allocated_PKR"] / 1e9
        fig_scatter = px.scatter(
            reg_comp,
            x="Allocated_B",
            y="Avg_Risk",
            size="Flagged_Projects",
            color="Flagged_Rate_%",
            text="region",
            color_continuous_scale="OrRd",
            title="Total Allocated Funds vs Average Risk Score",
            labels={
                "Allocated_B": "Total Allocated (PKR Billions)",
                "Avg_Risk": "Average Risk Score (0–100)",
                "Flagged_Rate_%": "Flagged %",
            },
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, width="stretch")

    with c2:
        dept_reg = (
            filtered_df.groupby(["region", "department"])["risk_score"]
            .mean()
            .reset_index()
            .pivot(index="region", columns="department", values="risk_score")
        )
        fig_heat = px.imshow(
            dept_reg,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Heatmap: Avg Risk Score (Region x Department)",
            labels={"color": "Avg Risk"},
        )
        st.plotly_chart(fig_heat, width="stretch")

    display_reg = reg_comp.copy()
    display_reg["Allocated"] = display_reg["Allocated_PKR"].apply(format_pkr)
    display_reg["Spent"] = display_reg["Spent_PKR"].apply(format_pkr)
    display_reg["Funds at Risk"] = display_reg["Funds_At_Risk_PKR"].apply(format_pkr)
    display_reg["Avg Risk Score"] = display_reg["Avg_Risk"].round(1)
    st.dataframe(
        display_reg[
            ["region", "Total_Projects", "Flagged_Projects", "Flagged_Rate_%", "Allocated", "Spent", "Funds at Risk", "Avg Risk Score"]
        ].sort_values("Avg Risk Score", ascending=False),
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------------------------
# TAB 4: CONTRACTOR WATCH
# ------------------------------------------------------------------------------
with tab_contractor:
    st.subheader("Contractor Risk & Market Concentration Watchlist")

    min_contracts = st.slider("Minimum Projects Won by Contractor", min_value=1, max_value=10, value=2)

    ctr_agg = (
        filtered_df.groupby("contractor_id")
        .agg(
            projects=("project_id", "count"),
            flagged_projects=("is_flagged", "sum"),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            total_allocated=("allocated_amount_pkr", "sum"),
            max_district_share=("contractor_concentration", "max"),
            single_source_count=("bid_type", lambda s: (s == "single source").sum()),
            total_audit_objections=("audit_objections_count", "sum"),
        )
        .reset_index()
    )
    ctr_filtered = ctr_agg[ctr_agg["projects"] >= min_contracts].copy()
    total_filtered_value = max(float(filtered_df["allocated_amount_pkr"].sum()), 1.0)
    ctr_filtered["overall_value_share_%"] = (ctr_filtered["total_allocated"] / total_filtered_value * 100.0).round(2)

    if ctr_filtered.empty:
        st.info("No contractors meet the selected minimum project count under current filters.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            top_risk_ctr = ctr_filtered.sort_values("avg_risk", ascending=False).head(15)
            fig_ctr_risk = px.bar(
                top_risk_ctr.sort_values("avg_risk"),
                x="avg_risk",
                y="contractor_id",
                orientation="h",
                color="avg_risk",
                color_continuous_scale=["#F59E0B", "#EA580C", "#DC2626"],
                hover_data={"projects": True, "flagged_projects": True, "single_source_count": True},
                title="Top 15 Contractors by Average Risk Score",
                labels={"avg_risk": "Avg Risk Score", "contractor_id": "Contractor ID"},
            )
            fig_ctr_risk.update_layout(coloraxis_showscale=False, height=450)
            st.plotly_chart(fig_ctr_risk, width="stretch")

        with col_c2:
            top_share_ctr = ctr_filtered.sort_values("total_allocated", ascending=False).head(15)
            top_share_ctr["Allocated (M PKR)"] = (top_share_ctr["total_allocated"] / 1e6).round(1)
            fig_ctr_share = px.bar(
                top_share_ctr.sort_values("Allocated (M PKR)"),
                x="Allocated (M PKR)",
                y="contractor_id",
                orientation="h",
                color="avg_risk",
                color_continuous_scale=["#10B981", "#F59E0B", "#DC2626"],
                hover_data={"projects": True, "overall_value_share_%": True, "max_district_share": ":.1%"},
                title="Top 15 Contractors by Total Contract Value (PKR Millions)",
                labels={"contractor_id": "Contractor ID", "avg_risk": "Avg Risk"},
            )
            fig_ctr_share.update_layout(height=450)
            st.plotly_chart(fig_ctr_share, width="stretch")

        ctr_table = ctr_filtered.sort_values(["avg_risk", "total_allocated"], ascending=False).copy()
        ctr_table["Total Contract Value"] = ctr_table["total_allocated"].apply(format_pkr)
        ctr_table["Max District Share %"] = (ctr_table["max_district_share"] * 100.0).round(1)
        ctr_table["Avg Risk"] = ctr_table["avg_risk"].round(1)

        st.dataframe(
            ctr_table[
                [
                    "contractor_id",
                    "projects",
                    "flagged_projects",
                    "Avg Risk",
                    "max_risk",
                    "Total Contract Value",
                    "overall_value_share_%",
                    "Max District Share %",
                    "single_source_count",
                    "total_audit_objections",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


# ------------------------------------------------------------------------------
# TAB 5: PROJECT EXPLORER & GEMINI AI ASSISTANT
# ------------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Project Explorer & Plain-English Flag Explanations")

    f1, f2, f3 = st.columns([1.5, 1.2, 1.0])
    with f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with f2:
        search_term = st.text_input(
            "Search Project ID, District, or Contractor",
            placeholder="e.g., Quetta, CTR-PK-091, PK-2425...",
        ).strip().lower()
    with f3:
        sort_by = st.selectbox(
            "Sort By",
            options=["risk_score", "allocated_amount_pkr", "spent_amount_pkr", "audit_objections_count"],
            index=0,
        )

    exp_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)].copy()
    if search_term:
        mask = (
            exp_df["project_id"].str.lower().str.contains(search_term, na=False)
            | exp_df["district"].str.lower().str.contains(search_term, na=False)
            | exp_df["contractor_id"].str.lower().str.contains(search_term, na=False)
            | exp_df["risk_explanation"].str.lower().str.contains(search_term, na=False)
        )
        exp_df = exp_df[mask]

    exp_df = exp_df.sort_values(sort_by, ascending=False)

    st.caption(f"Showing **{len(exp_df):,}** projects matching explorer criteria.")

    display_cols = [
        "project_id",
        "risk_score",
        "risk_level",
        "region",
        "district",
        "department",
        "fiscal_year",
        "allocated_amount_pkr",
        "released_amount_pkr",
        "spent_amount_pkr",
        "completion_percent",
        "contractor_id",
        "bid_type",
        "bidders_count",
        "audit_objections_count",
        "risk_explanation",
    ]

    st.dataframe(
        exp_df[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "allocated_amount_pkr": st.column_config.NumberColumn("Allocated (PKR)", format="%,.0f"),
            "released_amount_pkr": st.column_config.NumberColumn("Released (PKR)", format="%,.0f"),
            "spent_amount_pkr": st.column_config.NumberColumn("Spent (PKR)", format="%,.0f"),
            "completion_percent": st.column_config.NumberColumn("Completion %", format="%.1f%%"),
            "risk_explanation": st.column_config.TextColumn("Plain-English Explanation", width="large"),
        },
    )

    exp_csv = exp_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Current Explorer Selection as CSV",
        data=exp_csv,
        file_name="flagged_projects_export.csv",
        mime="text/csv",
    )

    st.divider()

    # Optional Gemini AI Bilingual Assistant
    st.subheader("🤖 AI Audit Assistant (English & Urdu Summary)")
    st.caption("Uses the Gemini API to translate technical risk flags into simple English and Urdu for auditors and citizens.")

    if exp_df.empty:
        st.info("Adjust the Project Explorer filters above so at least one project is available to summarize.")
    else:
        selected_pid = st.selectbox(
            "Select a Project to Summarize",
            options=exp_df["project_id"].tolist(),
            index=0,
        )
        proj = exp_df[exp_df["project_id"] == selected_pid].iloc[0]

        with st.container(border=True):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Risk Score", f"{proj['risk_score']:.1f} / 100", proj["risk_level"])
            p2.metric("Allocated", format_pkr(proj["allocated_amount_pkr"]))
            p3.metric("Spent", format_pkr(proj["spent_amount_pkr"]))
            p4.metric("Completion", f"{proj['completion_percent']}%")
            st.markdown(f"**Detected Risk Explanation:** {proj['risk_explanation']}")

        api_key = get_gemini_api_key(manual_gemini_key)
        if not api_key:
            st.info(
                "No Gemini API key found. To enable live AI summaries, add `GEMINI_API_KEY = \"your-key\"` "
                "in Streamlit Secrets (`.streamlit/secrets.toml`) or paste a temporary key in the sidebar."
            )
        else:
            if st.button("✨ Summarize Selected Project in English & Urdu", type="primary"):
                with st.spinner("Generating bilingual audit brief with Gemini..."):
                    try:
                        summary_md = generate_bilingual_summary(proj, api_key)
                        st.markdown(summary_md)
                    except Exception as exc:
                        st.error(f"Gemini API call failed: {exc}")


# ------------------------------------------------------------------------------
# TAB 6: BENFORD'S LAW ANALYSIS
# ------------------------------------------------------------------------------
with tab_benford:
    st.subheader("Benford's Law First-Digit Forensic Check")
    st.markdown(
        """
        In natural financial datasets spanning orders of magnitude, leading digits follow **Benford's Law**
        (~30.1% start with `1`, falling to ~4.6% starting with `9`). Fabricated invoices, threshold splitting
        (e.g., PKR 4.9M / 9.9M / 49.5M), or arbitrary round-number entries cause spikes on specific digits.
        """
    )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        benford_region_opts = ["All Filtered Regions"] + sorted(filtered_df["region"].unique().tolist())
        benford_region = st.selectbox("Select Region for Digit Distribution", options=benford_region_opts)
    with b_col2:
        benford_metric = st.selectbox(
            "Amount Field to Analyze",
            options=["spent_amount_pkr", "allocated_amount_pkr", "released_amount_pkr"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    b_subset = filtered_df if benford_region == "All Filtered Regions" else filtered_df[filtered_df["region"] == benford_region]
    b_df, mad, chi2_stat, p_val, conformity, n_obs = compute_benford_stats(b_subset[benford_metric])

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Records Analyzed", f"{n_obs:,}")
    bm2.metric("Mean Absolute Deviation (MAD)", f"{mad:.4f}")
    bm3.metric("Chi-Square p-value", f"{p_val:.4f}")
    bm4.metric("Benford Conformity", conformity)

    fig_benford = go.Figure()
    fig_benford.add_trace(
        go.Bar(
            x=b_df["Digit"],
            y=b_df["Actual (%)"],
            name="Actual Distribution (%)",
            marker_color="#2563EB",
        )
    )
    fig_benford.add_trace(
        go.Scatter(
            x=b_df["Digit"],
            y=b_df["Expected Benford (%)"],
            name="Expected Benford's Law (%)",
            mode="lines+markers",
            line=dict(color="#DC2626", width=3),
            marker=dict(size=8),
        )
    )
    fig_benford.update_layout(
        title=f"First-Digit Distribution vs Benford's Law — {benford_region}",
        xaxis=dict(title="Leading Digit (1–9)", tickmode="linear", dtick=1),
        yaxis_title="Share of Projects (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
    )
    st.plotly_chart(fig_benford, width="stretch")

    # Regional Benford Conformity Comparison
    reg_benford_rows = []
    for reg_name in sorted(filtered_df["region"].unique()):
        sub_s = filtered_df[filtered_df["region"] == reg_name][benford_metric]
        _, r_mad, _, r_p, r_conf, r_n = compute_benford_stats(sub_s)
        reg_benford_rows.append(
            {
                "Region": reg_name,
                "Projects": r_n,
                "MAD": round(r_mad, 4),
                "p-value": round(r_p, 4),
                "Conformity Status": r_conf,
            }
        )

    reg_benford_df = pd.DataFrame(reg_benford_rows).sort_values("MAD", ascending=False)

    col_b1, col_b2 = st.columns([1.2, 1.0])
    with col_b1:
        fig_mad = px.bar(
            reg_benford_df.sort_values("MAD"),
            x="MAD",
            y="Region",
            orientation="h",
            color="MAD",
            color_continuous_scale=["#10B981", "#F59E0B", "#DC2626"],
            title="Benford Deviation (MAD) by Region (Higher = Larger Deviation)",
        )
        fig_mad.add_vline(x=0.015, line_dash="dash", line_color="#DC2626", annotation_text="Nonconformity (0.015)")
        fig_mad.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_mad, width="stretch")
    with col_b2:
        st.markdown("#### Regional Benford Summary")
        st.dataframe(reg_benford_df, width="stretch", hide_index=True)
