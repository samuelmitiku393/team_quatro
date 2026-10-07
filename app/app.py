# app.py
"""Streamlit Portfolio Showcase: Addis Ababa Ride Demand Forecaster.

Developed by Team Quatro (Led by Samuel Mitiku).
An enterprise-grade operational dispatch and machine learning platform predicting
hourly ride demand across 12 zones in Addis Ababa using a tuned LightGBM regressor.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Addis Mobility AI | Team Quatro (Led by Samuel Mitiku)",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Portfolio-Grade Aesthetics)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Header */
    .hero-container {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.92) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }

    /* Badges */
    .badge-chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .badge-chip-green {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-chip-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    /* Metric Cards */
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 0.35rem;
    }

    /* Sidebar Author Profile */
    .author-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.2rem;
    }
    .author-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.1rem;
    }
    .author-role {
        font-size: 0.8rem;
        color: #38bdf8;
        font-weight: 500;
        margin-bottom: 0.6rem;
    }
    .author-bio {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.4;
        margin-bottom: 0.8rem;
    }

    /* Insight and Card Containers */
    .card-box {
        padding: 1.25rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# ASSET PATH RESOLUTION & CONSTANTS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
if not ASSETS_DIR.exists():
    ASSETS_DIR = Path("assets")

FIGURES_DIR = ASSETS_DIR / "figures"
if not FIGURES_DIR.exists():
    FIGURES_DIR = BASE_DIR.parent / "figures"

MIN_DATE = datetime.date(2025, 11, 1)
MAX_DATE = datetime.date(2025, 11, 14)

ZONE_MAP = {
    "Arat Kilo": "ARAT KILO",
    "Ayat": "AYAT",
    "Bole": "BOLE",
    "CMC": "CMC",
    "Gerji": "GERJI",
    "Kazanchis": "KAZANCHIS",
    "Kolfe Keranio": "KOLFE",
    "Lideta": "LIDETA",
    "Megenagna": "MEGENAGNA",
    "Merkato": "MERKATO",
    "Piassa": "PIASSA",
    "Sarbet": "SARBET",
}
DISPLAY_ZONES = list(ZONE_MAP.keys())

# -----------------------------------------------------------------------------
# DATA LOADING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_forecast_dataset() -> pd.DataFrame:
    """Load the pre-joined LightGBM competition test predictions and metadata."""
    csv_path = ASSETS_DIR / "forecast_predictions.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
        return df

    # Fallback to parent master test & submission if running from repo root
    parent_test = BASE_DIR.parent / "data" / "processed" / "master_test.csv"
    parent_sub = BASE_DIR.parent / "submission" / "team_quatro_submission.csv"
    if parent_test.exists() and parent_sub.exists():
        t_df = pd.read_csv(parent_test)
        s_df = pd.read_csv(parent_sub)
        merged = pd.merge(t_df, s_df, on="row_id")
        merged["pickup_hour"] = pd.to_datetime(merged["pickup_hour"])
        return merged

    raise FileNotFoundError("Forecast predictions dataset not found. Please verify assets.")


@st.cache_data
def load_events_metadata() -> pd.DataFrame:
    """Load the scheduled city events calendar."""
    events_path = ASSETS_DIR / "events_calendar.csv"
    if not events_path.exists():
        events_path = BASE_DIR.parent / "data" / "raw" / "events_calendar.csv"
    if events_path.exists():
        df = pd.read_csv(events_path)
        df["start_dt"] = pd.to_datetime(df["start_datetime"], errors="coerce")
        df["end_dt"] = pd.to_datetime(df["end_datetime"], errors="coerce")
        return df
    return pd.DataFrame()


# Load datasets
try:
    ALL_PREDICTIONS = load_forecast_dataset()
    EVENTS_DF = load_events_metadata()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR: AUTHOR BIO & FORECAST CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="author-card">
            <div class="author-name">Samuel Mitiku</div>
            <div class="author-role">Project & Team Lead | Team Quatro</div>
            <div class="author-bio">
                Lead Data Scientist & ML Systems Architect. Speared end-to-end time-series modeling,
                leakage-proof cross-validation, and operational ride dispatching for Addis Ababa.
            </div>
            <div style="display: flex; gap: 8px;">
                <a href="https://github.com/dagix7/team_quatro" target="_blank" style="text-decoration: none;">
                    <span class="badge-chip" style="margin:0; cursor:pointer;">💻 GitHub Repo</span>
                </a>
                <a href="https://linkedin.com" target="_blank" style="text-decoration: none;">
                    <span class="badge-chip-green" style="margin:0; cursor:pointer;">👔 LinkedIn</span>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🕹️ Operational Controls")
    selected_zone_display = st.selectbox("Select Zone (12 Zones)", DISPLAY_ZONES, index=2)
    selected_db_zone = ZONE_MAP[selected_zone_display]

    selected_date = st.date_input(
        "Forecast Horizon Date",
        value=MIN_DATE,
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        help="Test period dates: November 1 to November 14, 2025",
    )

    st.markdown("---")
    st.subheader("⚙️ Fleet & Fare Parameters")
    trips_per_driver = st.slider(
        "Driver Productivity (trips/hr)",
        min_value=0.8,
        max_value=2.2,
        value=1.3,
        step=0.1,
        help="Standard benchmark: ~1.3 trips per driver hour in Addis Ababa traffic.",
    )
    fare_per_trip = st.number_input(
        "Average Trip Fare (ETB)",
        min_value=100,
        max_value=800,
        value=250,
        step=25,
        help="Historical average fare per ride request.",
    )

    st.markdown("---")
    st.subheader("🧪 Scenario Stress-Testing")
    apply_surge = st.checkbox("Simulate Severe Weather Shock (+25% Surge)", value=False)
    surge_multiplier = 1.25 if apply_surge else 1.0


# -----------------------------------------------------------------------------
# FILTERING CURRENT 24-HOUR FORECAST SLICE
# -----------------------------------------------------------------------------
mask = (
    (ALL_PREDICTIONS["zone"] == selected_db_zone)
    & (ALL_PREDICTIONS["pickup_hour"].dt.date == selected_date)
)
day_forecast = ALL_PREDICTIONS[mask].copy().sort_values("pickup_hour")

if day_forecast.empty:
    st.warning("No records found for the selected zone and date combination.")
    st.stop()

# Apply any scenario stress testing multiplier
day_forecast["effective_trips"] = day_forecast["predicted_trips"] * surge_multiplier
day_forecast["hour"] = day_forecast["pickup_hour"].dt.hour
day_forecast["hour_label"] = day_forecast["hour"].apply(lambda h: f"{h:02d}:00")

# Summary aggregates
total_trips = day_forecast["effective_trips"].sum()
peak_idx = day_forecast["effective_trips"].idxmax()
peak_hour = int(day_forecast.loc[peak_idx, "hour"])
peak_trips = float(day_forecast.loc[peak_idx, "effective_trips"])
recommended_drivers = int(round(total_trips / trips_per_driver))
projected_revenue = int(round(total_trips * fare_per_trip))

# Silent Weather & Events lookup
total_rain_mm = day_forecast["rain_mm"].sum()
max_temp = day_forecast["temp_c"].max()
min_temp = day_forecast["temp_c"].min()

day_events = []
if not EVENTS_DF.empty:
    for _, row in EVENTS_DF.iterrows():
        r_zone = str(row.get("zone", "")).upper()
        s_date = row["start_dt"].date() if pd.notna(row["start_dt"]) else None
        e_date = row["end_dt"].date() if pd.notna(row["end_dt"]) else s_date
        is_match = ("CITYWIDE" in r_zone or "ALL" in r_zone or selected_db_zone in r_zone)
        if is_match and s_date and e_date and (s_date <= selected_date <= e_date):
            day_events.append(f"{row.get('event_name', 'Event')} ({row.get('event_type', 'public')})")


# -----------------------------------------------------------------------------
# HERO HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">🚕 Addis Ababa Ride Demand AI</div>
        <div class="hero-subtitle">
            Enterprise Machine Learning & Operational Dispatch System &nbsp;|&nbsp; 
            <strong>Team Quatro</strong> (Led by Samuel Mitiku)
        </div>
        <div>
            <span class="badge-chip">🏆 Competition Finalist</span>
            <span class="badge-chip-green">⚡ LightGBM Regressor (RMSE: 15.79 vs 30.16 Baseline)</span>
            <span class="badge-chip-purple">📍 12 Addis Ababa Zones</span>
            <span class="badge-chip">🔒 Zero Leakage Chronological CV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# APPLICATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Operational Forecaster",
    "🧠 ML Architecture & Validation",
    "📊 EDA & Spatial-Temporal Insights",
    "🏗️ Pipeline & About the Author",
])

# =============================================================================
# TAB 1: LIVE OPERATIONAL FORECASTER
# =============================================================================
with tab1:
    st.markdown(
        f"### 📍 Dispatch Overview: **{selected_zone_display}** on **{selected_date.strftime('%A, %B %d, %Y')}**"
    )

    # 4 Key Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">⏰ Peak Demand Window</div>
                <div class="metric-value">{peak_hour:02d}:00</div>
                <div class="metric-sub">{peak_trips:.1f} trips at peak hour</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">👥 Recommended Fleet</div>
                <div class="metric-value">{recommended_drivers:,}</div>
                <div class="metric-sub">Across 24h ({total_trips:,.0f} total trips)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">💰 Projected Revenue</div>
                <div class="metric-value">{projected_revenue:,.0f} <span style="font-size:1rem;color:#94a3b8;">ETB</span></div>
                <div class="metric-sub">{fare_per_trip} ETB avg fare/trip</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        weather_status = f"🌧️ {total_rain_mm:.1f} mm Rain" if total_rain_mm > 0 else "☀️ Clear / Dry"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🌤️ Environmental Outlook</div>
                <div class="metric-value" style="font-size:1.45rem;">{weather_status}</div>
                <div class="metric-sub">Temp: {min_temp:.1f}°C – {max_temp:.1f}°C (EAT)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive 24-Hour Demand Curve
    st.subheader("📈 24-Hour Calibrated Demand Forecast Curve")

    chart_df = day_forecast.copy()

    # Altair Chart Definition
    base = alt.Chart(chart_df).encode(
        x=alt.X("hour:Q", title="Hour of Day (00:00 – 23:00 EAT)", scale=alt.Scale(domain=[0, 23])),
    )

    area = base.mark_area(
        line={"color": "#38bdf8", "width": 2.5},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(56, 189, 248, 0.45)", offset=0),
                alt.GradientStop(color="rgba(56, 189, 248, 0.02)", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        y=alt.Y("effective_trips:Q", title="Projected Hourly Ride Requests"),
        tooltip=[
            alt.Tooltip("hour_label:N", title="Local Time"),
            alt.Tooltip("effective_trips:Q", title="Predicted Trips", format=".1f"),
            alt.Tooltip("temp_c:Q", title="Temperature (°C)", format=".1f"),
            alt.Tooltip("rain_mm:Q", title="Rainfall (mm)", format=".1f"),
        ],
    )

    points = base.mark_circle(size=60, color="#38bdf8").encode(
        y=alt.Y("effective_trips:Q"),
        tooltip=[
            alt.Tooltip("hour_label:N", title="Local Time"),
            alt.Tooltip("effective_trips:Q", title="Predicted Trips", format=".1f"),
            alt.Tooltip("temp_c:Q", title="Temperature (°C)", format=".1f"),
            alt.Tooltip("rain_mm:Q", title="Rainfall (mm)", format=".1f"),
        ],
    )

    # Peak Annotation
    peak_row = pd.DataFrame([{
        "hour": peak_hour,
        "effective_trips": peak_trips,
        "label": f"PEAK: {peak_trips:.1f} Trips ({peak_hour:02d}:00)",
    }])
    peak_rule = alt.Chart(peak_row).mark_rule(color="#ef4444", strokeDash=[4, 4], strokeWidth=1.5).encode(x="hour:Q")
    peak_marker = alt.Chart(peak_row).mark_point(color="#ef4444", size=220, shape="star", filled=True).encode(
        x="hour:Q", y="effective_trips:Q"
    )
    peak_text = alt.Chart(peak_row).mark_text(
        align="left", dx=8, dy=-14, color="#ef4444", fontWeight=700, fontSize=13
    ).encode(x="hour:Q", y="effective_trips:Q", text="label")

    demand_chart = (area + points + peak_rule + peak_marker + peak_text).properties(
        height=380
    ).interactive()

    st.altair_chart(demand_chart, use_container_width=True)

    # Silent Context Bar
    events_str = "; ".join(day_events) if day_events else "No major city events scheduled in this zone."
    rain_str = (
        f"Precipitation expected: {total_rain_mm:.1f} mm across the day."
        if total_rain_mm > 0
        else "Dry conditions with zero precipitation forecast."
    )
    st.info(
        f"ℹ️ **Silent Background Context Lookups:**  \n"
        f"• **Weather:** {rain_str} Temperature ranges from {min_temp:.1f}°C to {max_temp:.1f}°C.  \n"
        f"• **Scheduled Events ({selected_zone_display}):** {events_str}"
    )

    # Hourly Dispatch Schedule Table
    with st.expander("📋 View & Export Full 24-Hour Dispatch Plan", expanded=False):
        table_df = pd.DataFrame({
            "Hour (EAT)": day_forecast["hour_label"],
            "Forecasted Trips": day_forecast["effective_trips"].round(1),
            "Drivers Needed": (day_forecast["effective_trips"] / trips_per_driver).round(1),
            "Expected Revenue (ETB)": (day_forecast["effective_trips"] * fare_per_trip).round(0).astype(int),
            "Temp (°C)": day_forecast["temp_c"].round(1),
            "Rain (mm)": day_forecast["rain_mm"].round(1),
            "Rain Surge Active": day_forecast["is_rainy"].apply(lambda x: "🌧️ Yes" if x > 0 else "☀️ No"),
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        csv_data = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download 24-Hour Dispatch Plan (CSV)",
            data=csv_data,
            file_name=f"dispatch_plan_{selected_db_zone}_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 2: ML ARCHITECTURE & MODEL EVALUATION
# =============================================================================
with tab2:
    st.markdown("### 🧠 Machine Learning Rigor & Model Scoreboard")
    st.markdown(
        "A rigorous, leakage-proof time-series benchmark evaluating baselines, linear models, and gradient boosting "
        "on the **October 2025 strict chronological holdout (8,690 zone-hours)**."
    )

    col_l1, col_l2 = st.columns([3, 2])

    with col_l1:
        st.markdown("#### 🏆 Holdout Performance Comparison")
        leaderboard_df = pd.DataFrame([
            {"Model Family": "LightGBM (Selected Champion)", "RMSE": 15.72, "MAE": 6.78, "Fit Time (s)": 2.28, "Status": "⭐️ Final Model"},
            {"Model Family": "Random Forest Regressor", "RMSE": 15.78, "MAE": 6.58, "Fit Time (s)": 25.24, "Status": "Competitive"},
            {"Model Family": "HistGradientBoosting", "RMSE": 15.78, "MAE": 6.88, "Fit Time (s)": 3.87, "Status": "Competitive"},
            {"Model Family": "Seasonal Naive (Zone × Hour × Day)", "RMSE": 17.16, "MAE": 7.53, "Fit Time (s)": 0.01, "Status": "Domain Baseline"},
            {"Model Family": "Ridge Regression (Linear)", "RMSE": 25.44, "MAE": 16.87, "Fit Time (s)": 0.21, "Status": "Linear Baseline"},
            {"Model Family": "Global Mean Baseline", "RMSE": 30.16, "MAE": 20.79, "Fit Time (s)": 0.01, "Status": "Naive Baseline"},
        ])
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        st.caption(
            "💡 **Key Finding:** LightGBM achieves a **47.9% reduction in RMSE** compared to the global mean "
            "and outpaces Seasonal Naive by ~1.44 RMSE points, while training in just **2.28 seconds**."
        )

    with col_l2:
        st.markdown("#### 📉 RMSE Error Reduction")
        bars = (
            alt.Chart(leaderboard_df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("RMSE:Q", title="Validation RMSE (Lower is Better)"),
                y=alt.Y("Model Family:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum["Model Family"] == "LightGBM (Selected Champion)",
                    alt.value("#38bdf8"),
                    alt.value("#64748b"),
                ),
                tooltip=["Model Family", "RMSE", "MAE", "Fit Time (s)"],
            )
            .properties(height=260)
        )
        st.altair_chart(bars, use_container_width=True)

    st.markdown("---")

    # Rolling-Origin CV & Ablation Study
    c_cv1, c_cv2 = st.columns(2)

    with c_cv1:
        st.markdown("#### 🔄 Rolling-Origin Temporal Cross-Validation")
        st.markdown(
            "To prove generalizability across seasonal shifts, we evaluated LightGBM against Seasonal Naive "
            "across **4 rolling 14-day temporal test folds** without any future leakage:"
        )
        cv_data = pd.DataFrame([
            {"Fold": "Fold 1", "Window": "Aug 18 – Aug 31", "Train Rows": 62057, "LightGBM RMSE": 10.97, "Seasonal Naive RMSE": 12.84},
            {"Fold": "Fold 2", "Window": "Sep 01 – Sep 14", "Train Rows": 65991, "LightGBM RMSE": 15.00, "Seasonal Naive RMSE": 17.97},
            {"Fold": "Fold 3", "Window": "Sep 15 – Sep 28", "Train Rows": 69925, "LightGBM RMSE": 12.94, "Seasonal Naive RMSE": 15.22},
            {"Fold": "Fold 4", "Window": "Sep 29 – Oct 12", "Train Rows": 73848, "LightGBM RMSE": 17.08, "Seasonal Naive RMSE": 18.04},
        ])
        st.dataframe(cv_data, use_container_width=True, hide_index=True)
        st.success("✅ **Cross-Validation Aggregate:** LightGBM scored **13.998 ± 2.633 RMSE**, outperforming the domain baseline on every single fold.")

    with c_cv2:
        st.markdown("#### 🧩 Feature Group Ablation Study")
        st.markdown(
            "Every feature group was isolated on the identical October holdout to measure real operational lift:"
        )
        ablation_df = pd.DataFrame([
            {"Feature Group": "Calendar + Zone + Trend", "Features": 8, "RMSE": 16.92, "Marginal Lift": "Base"},
            {"Feature Group": "+ Hourly Weather Features", "Features": 15, "RMSE": 15.99, "Marginal Lift": "-0.93 RMSE (Significant)"},
            {"Feature Group": "+ Asymmetric Event Windows", "Features": 21, "RMSE": 16.72, "Marginal Lift": "-0.20 RMSE"},
            {"Feature Group": "+ Combined (Weather + Events)", "Features": 28, "RMSE": 15.72, "Marginal Lift": "⭐️ -1.20 RMSE Net Gain"},
        ])
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
        st.info("💡 **Ablation Takeaway:** Weather integration provides an essential -0.93 RMSE boost, while event lead/lag windows capture acute spikes.")

    st.markdown("---")

    # Leakage Audit Callout
    st.markdown("#### 🛡️ Data Leakage & Production Safety Audit")
    st.markdown(
        """
        In production ride-hailing dispatch, features like `avg_wait_min`, `active_drivers`, and `avg_fare_birr`
        are **consequences of demand**, not precursors.
        - **The Pitfall:** Training on these operational variables produced an artificial holdout score of **13.55 RMSE** (a 19.9% illusory boost).
        - **The Fix:** Under Samuel Mitiku's guidance, all post-request variables were strictly audited and barred from feature tables, guaranteeing that the model relies solely on features known prior to dispatch.
        """
    )


# =============================================================================
# TAB 3: EDA & SPATIAL-TEMPORAL INSIGHTS
# =============================================================================
with tab3:
    st.markdown("### 📊 Exploratory Spatial & Temporal Insights")
    st.markdown(
        "Key empirical discoveries uncovered during exploratory analysis that directly influenced our feature engineering."
    )

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("#### 1. ☀️ The UTC to EAT (+3h) Timezone Discovery")
        fig06_path = FIGURES_DIR / "fig06_weather_timezone_check.png"
        if fig06_path.exists():
            st.image(str(fig06_path), caption="Solar peak aligns with 14:00 only after +3h shift to East Africa Time", use_container_width=True)
        else:
            st.info("Solar noon temperature peak shifted by +3h to resolve raw sensor UTC offsets.")
        st.caption(
            "**Takeaway:** Raw temperature logs peaked around 11:00 UTC. Plotting physical diurnal solar peaks "
            "proved that raw weather was recorded in UTC, requiring an explicit conversion to EAT (+3h) to align with commuter traffic."
        )

    with row1_c2:
        st.markdown("#### 2. 🌧️ Non-Linear Rain Surge & Saturation")
        fig07_path = FIGURES_DIR / "fig07_rain_effect.png"
        if fig07_path.exists():
            st.image(str(fig07_path), caption="Demand lift across rainfall intensity bins relative to dry conditions", use_container_width=True)
        else:
            st.info("Rainfall creates a sharp demand uplift in light/moderate conditions.")
        st.caption(
            "**Takeaway:** Commuters actively seek ride-hailing shelter during light to moderate rain (+15–30% surge), "
            "whereas extreme precipitation causes saturation and travel cancellation."
        )

    st.markdown("---")

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("#### 3. 🏟️ Asymmetric Event Dynamics (-2h / +2h)")
        fig08_path = FIGURES_DIR / "fig08_event_study.png"
        if fig08_path.exists():
            st.image(str(fig08_path), caption="Surge waves surrounding football matches and major concerts", use_container_width=True)
        else:
            st.info("Events generate a massive 1.55x surge in the 2 hours immediately following conclusion.")
        st.caption(
            "**Takeaway:** Pre-event arrival is gradual over 3 hours, but post-event exit creates an acute **1.55x surge** "
            "compressed into the 2 hours following event completion, justifying our asymmetric lead-lag features."
        )

    with row2_c2:
        st.markdown("#### 4. 🏙️ Commercial vs. Residential Diurnal Curves")
        fig04_path = FIGURES_DIR / "fig04_hour_by_weekday_heatmap.png"
        if fig04_path.exists():
            st.image(str(fig04_path), caption="Weekday vs Weekend demand heatmaps across Addis Ababa", use_container_width=True)
        else:
            st.info("Weekday commute surges vs weekend late-night Bole airport & nightlife demand.")
        st.caption(
            "**Takeaway:** Commuter zones (Merkato, Piassa) experience sharp 08:00 and 18:00 spikes, "
            "whereas Bole maintains strong late-night weekend demand due to international travel and hospitality."
        )


# =============================================================================
# TAB 4: SYSTEM ARCHITECTURE & ABOUT THE AUTHOR
# =============================================================================
with tab4:
    st.markdown("### 🏗️ End-to-End System Pipeline & Leadership")

    col_arch1, col_arch2 = st.columns([3, 2])

    with col_arch1:
        st.markdown("#### 🔄 Production Pipeline Architecture")
        st.markdown(
            """
            ```text
            [ Raw Trip Logs (12 Zones) ]     [ Hourly Weather API ]     [ City Events Calendar ]
                         │                                │                            │
                         ▼                                ▼                            ▼
            [ Timestamp Normalization ]      [ UTC -> EAT (+3h) Shift ]    [ Asymmetric Windows ]
                         │                                │                            │
                         └─────────────────┬──────────────┴────────────────────────────┘
                                           │
                                           ▼
                            [ Master Feature Store (28 Features) ]
                            • Diurnal Cyclical (Hour, Day, Weekday)
                            • Weather Integrations (Temp, Rain 3h Lag)
                            • Event Windows (Active, Lead, Lag)
                            • Categorical Zone Embeddings
                                           │
                                           ▼
                            [ Time-Series Cross Validation ]
                            • Rolling-Origin 14-Day Windows
                            • Strict Zero-Leakage Enforcement
                                           │
                                           ▼
                            [ Tuned LightGBM Regressor (RMSE: 15.72) ]
                                           │
                                           ▼
                            [ Streamlit Operational Dispatch Console ]
                            • 24h Hourly Demand Curves & Fleet Sizing
                            • Gross Fare Revenue Projection (ETB)
                            • Dynamic Weather & Scenario Stress Testing
            ```
            """
        )

        st.markdown("#### 🛠️ Technology Stack")
        st.markdown(
            """
            - **Modeling & ML:** `LightGBM`, `Scikit-Learn`, `Joblib`
            - **Data Engineering:** `Pandas`, `NumPy`, `Datetime`
            - **Visualization & UI:** `Streamlit`, `Altair`, `Plotly`, `Matplotlib`
            - **Deployment:** Streamlit Cloud Ready, Docker Compatible
            """
        )

    with col_arch2:
        st.markdown("#### 👤 About the Author & Project Lead")
        st.markdown(
            """
            <div class="card-box" style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(56, 189, 248, 0.3);">
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;">
                    Samuel Mitiku
                </div>
                <div style="font-size: 0.9rem; color: #38bdf8; font-weight: 600; margin-bottom: 0.9rem;">
                    Lead Data Scientist & ML Systems Architect
                </div>
                <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                    Led <strong>Team Quatro</strong> in architecting an end-to-end urban mobility forecasting platform.
                    Specialized in time-series forecasting, automated feature engineering, leakage prevention,
                    and operationalizing machine learning models into intuitive decision-support systems.
                </p>
                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 0.8rem 0;">
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem;">
                    <strong>Core Strengths Demonstrated:</strong>
                    <ul style="padding-left: 1.2rem; margin-top: 0.4rem;">
                        <li>Production-grade Time Series Forecasting</li>
                        <li>High-throughput LightGBM Optimization</li>
                        <li>Robust Data Cleaning & Sensor Alignment</li>
                        <li>Executive Dashboard Design & Full-stack ML</li>
                    </ul>
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="https://github.com/dagix7/team_quatro" target="_blank" style="text-decoration:none;">
                        <span class="badge-chip" style="cursor:pointer;">💻 GitHub Project</span>
                    </a>
                    <a href="https://linkedin.com" target="_blank" style="text-decoration:none;">
                        <span class="badge-chip-green" style="cursor:pointer;">🤝 Connect on LinkedIn</span>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 1rem 0;">
        <strong>Addis Ababa Ride Demand Forecaster</strong> &nbsp;|&nbsp; 
        Developed by <strong>Team Quatro</strong> (Led by Samuel Mitiku) &nbsp;|&nbsp;
        Powered by Streamlit & LightGBM &nbsp;|&nbsp; 
        <a href="https://github.com/dagix7/team_quatro" target="_blank" style="color: #38bdf8; text-decoration: none;">View Source Code</a>
    </div>
    """,
    unsafe_allow_html=True,
)
