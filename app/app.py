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
    page_title="Addis Mobility AI — Team Quatro",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DESIGN SYSTEM — Professional Slate + Indigo Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-page:      #0f1117;
        --bg-surface:   #161b27;
        --bg-elevated:  #1c2333;
        --border:       #252d3d;
        --border-focus: #3b5bdb;
        --text-primary: #e8eaf0;
        --text-secondary: #8892a4;
        --text-muted:   #5a6478;
        --accent:       #4c6ef5;
        --accent-light: #748ffc;
        --accent-dim:   rgba(76, 110, 245, 0.12);
        --success:      #2f9e44;
        --success-dim:  rgba(47, 158, 68, 0.10);
        --warning:      #e67700;
        --warning-dim:  rgba(230, 119, 0, 0.10);
        --danger:       #c92a2a;
        --radius-sm:    6px;
        --radius-md:    10px;
        --radius-lg:    14px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }

    /* ── Page background ── */
    .stApp { background-color: var(--bg-page); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Header Banner ── */
    .page-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        padding: 1.5rem 1.75rem;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        margin-bottom: 1.25rem;
        border-left: 4px solid var(--accent);
    }
    .page-header-left {}
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
    }
    .page-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0 0 0.8rem 0;
        line-height: 1.5;
    }
    .page-header-right {
        text-align: right;
        font-size: 0.78rem;
        color: var(--text-muted);
        flex-shrink: 0;
        padding-left: 1.5rem;
    }
    .page-header-right strong {
        color: var(--text-secondary);
        display: block;
        font-size: 0.82rem;
        margin-bottom: 2px;
    }

    /* ── Tag Pills ── */
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.73rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.25rem;
        letter-spacing: 0.01em;
    }
    .tag-blue  { background: var(--accent-dim);   color: var(--accent-light); border: 1px solid rgba(76,110,245,0.2); }
    .tag-green { background: var(--success-dim);  color: #69db7c;              border: 1px solid rgba(47,158,68,0.2); }
    .tag-amber { background: var(--warning-dim);  color: #ffd43b;              border: 1px solid rgba(230,119,0,0.2); }
    .tag-slate { background: rgba(88,101,122,0.12); color: var(--text-secondary); border: 1px solid var(--border); }

    /* ── KPI Cards ── */
    .kpi-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--accent);
        border-radius: 2px 0 0 2px;
    }
    .kpi-card.green::before { background: var(--success); }
    .kpi-card.amber::before { background: var(--warning); }
    .kpi-card.slate::before { background: #495057; }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        margin-bottom: 0.45rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }
    .kpi-unit {
        font-size: 0.95rem;
        font-weight: 400;
        color: var(--text-secondary);
        letter-spacing: 0;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.1rem;
    }

    /* ── Section Heading ── */
    .section-heading {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* ── Context Info Bar ── */
    .context-bar {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .context-bar strong { color: var(--text-primary); }

    /* ── Author Profile Card ── */
    .author-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--accent);
        padding: 1rem 1.1rem;
        margin-bottom: 1.25rem;
    }
    .author-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .author-role {
        font-size: 0.75rem;
        color: var(--accent-light);
        font-weight: 500;
        margin-bottom: 0.6rem;
    }
    .author-bio {
        font-size: 0.77rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-bottom: 0.75rem;
    }

    /* ── Profile Card (Tab 4) ── */
    .profile-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--accent);
        padding: 1.5rem;
    }
    .profile-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 3px;
        letter-spacing: -0.02em;
    }
    .profile-title {
        font-size: 0.82rem;
        color: var(--accent-light);
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .profile-bio {
        font-size: 0.84rem;
        color: var(--text-secondary);
        line-height: 1.65;
        margin-bottom: 1rem;
    }
    .profile-skills-title {
        font-size: 0.73rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .profile-skill-item {
        font-size: 0.81rem;
        color: var(--text-secondary);
        padding: 0.3rem 0;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .profile-skill-item:last-child { border-bottom: none; }

    /* ── Stat Row (Tab 2 hero numbers) ── */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .stat-item {
        flex: 1;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent-light);
        letter-spacing: -0.03em;
    }
    .stat-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.25rem 0;
    }

    /* ── Footer ── */
    .page-footer {
        text-align: center;
        padding: 1.25rem 0 0.75rem;
        font-size: 0.78rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border);
        margin-top: 0.5rem;
    }
    .page-footer a {
        color: var(--accent-light);
        text-decoration: none;
    }

    /* ── Link Buttons ── */
    .link-btn {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 0.3rem 0.7rem;
        border-radius: var(--radius-sm);
        font-size: 0.76rem;
        font-weight: 500;
        text-decoration: none;
        margin-right: 0.4rem;
        transition: opacity 0.15s;
    }
    .link-btn:hover { opacity: 0.8; }
    .link-btn-blue  { background: var(--accent-dim); color: var(--accent-light); border: 1px solid rgba(76,110,245,0.25); }
    .link-btn-green { background: var(--success-dim); color: #69db7c; border: 1px solid rgba(47,158,68,0.25); }

    /* ── Streamlit element overrides ── */
    div[data-testid="stMetric"]        { background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.75rem 1rem; }
    div[data-testid="stDataFrame"]     { border-radius: var(--radius-sm); overflow: hidden; }
    .stTabs [data-baseweb="tab-list"]  { background: var(--bg-surface); border-bottom: 1px solid var(--border); gap: 0; }
    .stTabs [data-baseweb="tab"]       { color: var(--text-secondary) !important; font-size: 0.85rem; font-weight: 500; padding: 0.6rem 1rem; border-radius: 0; }
    .stTabs [aria-selected="true"]     { color: var(--text-primary) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
    .stTabs [data-testid="stTabPanel"] { padding-top: 1.5rem; }
    .stExpander                        { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; background: var(--bg-surface) !important; }
    .stDownloadButton button           { border: 1px solid var(--border) !important; background: var(--bg-elevated) !important; color: var(--text-secondary) !important; font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# ASSET PATH RESOLUTION & CONSTANTS
# -----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
if not ASSETS_DIR.exists():
    ASSETS_DIR = Path("assets")

FIGURES_DIR = ASSETS_DIR / "figures"
if not FIGURES_DIR.exists():
    FIGURES_DIR = BASE_DIR.parent / "figures"

MIN_DATE = datetime.date(2025, 11, 1)
MAX_DATE = datetime.date(2025, 11, 14)

ZONE_MAP = {
    "Arat Kilo":    "ARAT KILO",
    "Ayat":         "AYAT",
    "Bole":         "BOLE",
    "CMC":          "CMC",
    "Gerji":        "GERJI",
    "Kazanchis":    "KAZANCHIS",
    "Kolfe Keranio":"KOLFE",
    "Lideta":       "LIDETA",
    "Megenagna":    "MEGENAGNA",
    "Merkato":      "MERKATO",
    "Piassa":       "PIASSA",
    "Sarbet":       "SARBET",
}
DISPLAY_ZONES = list(ZONE_MAP.keys())

# Chart color constants (matches CSS theme)
ACCENT      = "#4c6ef5"
ACCENT_LIGHT= "#748ffc"
SLATE       = "#3d4a5e"
DANGER      = "#fa5252"
SUCCESS     = "#51cf66"

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_forecast_dataset() -> pd.DataFrame:
    csv_path = ASSETS_DIR / "forecast_predictions.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
        return df
    parent_test = BASE_DIR.parent / "data" / "processed" / "master_test.csv"
    parent_sub  = BASE_DIR.parent / "submission" / "team_quatro_submission.csv"
    if parent_test.exists() and parent_sub.exists():
        merged = pd.merge(pd.read_csv(parent_test), pd.read_csv(parent_sub), on="row_id")
        merged["pickup_hour"] = pd.to_datetime(merged["pickup_hour"])
        return merged
    raise FileNotFoundError("Forecast predictions dataset not found.")


@st.cache_data
def load_events_metadata() -> pd.DataFrame:
    events_path = ASSETS_DIR / "events_calendar.csv"
    if not events_path.exists():
        events_path = BASE_DIR.parent / "data" / "raw" / "events_calendar.csv"
    if events_path.exists():
        df = pd.read_csv(events_path)
        df["start_dt"] = pd.to_datetime(df["start_datetime"], errors="coerce")
        df["end_dt"]   = pd.to_datetime(df["end_datetime"],   errors="coerce")
        return df
    return pd.DataFrame()


try:
    ALL_PREDICTIONS = load_forecast_dataset()
    EVENTS_DF       = load_events_metadata()
except Exception as e:
    st.error(f"Failed to load system data: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="author-card">
            <div class="author-name">Samuel Mitiku</div>
            <div class="author-role">Project Lead &amp; ML Systems Architect</div>
            <div class="author-bio">
                Led Team Quatro through end-to-end time-series forecasting,
                leakage-proof cross-validation, and operational dispatch
                system design for Addis Ababa's ride-hailing network.
            </div>
            <a href="https://github.com/samuelmitiku393/team_quatro" target="_blank" class="link-btn link-btn-blue">
                ↗ GitHub Repo
            </a>
            <a href="https://linkedin.com" target="_blank" class="link-btn link-btn-green">
                ↗ LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Forecast Controls**")
    selected_zone_display = st.selectbox("Zone", DISPLAY_ZONES, index=2, label_visibility="collapsed")
    selected_db_zone = ZONE_MAP[selected_zone_display]

    selected_date = st.date_input(
        "Date",
        value=MIN_DATE,
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        label_visibility="collapsed",
        help="Nov 1–14, 2025 test period.",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("**Fleet Parameters**")
    trips_per_driver = st.slider(
        "Driver productivity (trips/hr)", 0.8, 2.2, 1.3, 0.1,
        help="City benchmark: ~1.3 trips per driver hour."
    )
    fare_per_trip = st.number_input(
        "Avg fare per trip (ETB)", 100, 800, 250, 25,
        help="Historical average fare across all zones."
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("**Scenario Testing**")
    apply_surge = st.checkbox("Severe weather shock (+25%)", value=False)
    surge_multiplier = 1.25 if apply_surge else 1.0

# -----------------------------------------------------------------------------
# DATA SLICE FOR SELECTED ZONE + DATE
# -----------------------------------------------------------------------------
mask = (
    (ALL_PREDICTIONS["zone"] == selected_db_zone)
    & (ALL_PREDICTIONS["pickup_hour"].dt.date == selected_date)
)
day_forecast = ALL_PREDICTIONS[mask].copy().sort_values("pickup_hour")

if day_forecast.empty:
    st.warning("No data found for the selected zone and date. Try a different combination.")
    st.stop()

day_forecast["effective_trips"] = day_forecast["predicted_trips"] * surge_multiplier
day_forecast["hour"]            = day_forecast["pickup_hour"].dt.hour
day_forecast["hour_label"]      = day_forecast["hour"].apply(lambda h: f"{h:02d}:00")

total_trips         = day_forecast["effective_trips"].sum()
peak_idx            = day_forecast["effective_trips"].idxmax()
peak_hour           = int(day_forecast.loc[peak_idx, "hour"])
peak_trips          = float(day_forecast.loc[peak_idx, "effective_trips"])
recommended_drivers = int(round(total_trips / trips_per_driver))
projected_revenue   = int(round(total_trips * fare_per_trip))
total_rain_mm       = day_forecast["rain_mm"].sum()
max_temp            = day_forecast["temp_c"].max()
min_temp            = day_forecast["temp_c"].min()

day_events = []
if not EVENTS_DF.empty:
    for _, row in EVENTS_DF.iterrows():
        r_zone = str(row.get("zone", "")).upper()
        s_date = row["start_dt"].date() if pd.notna(row["start_dt"]) else None
        e_date = row["end_dt"].date()   if pd.notna(row["end_dt"])   else s_date
        if ("CITYWIDE" in r_zone or "ALL" in r_zone or selected_db_zone in r_zone):
            if s_date and e_date and (s_date <= selected_date <= e_date):
                day_events.append(
                    f"{row.get('event_name', 'Event')} — {row.get('event_type', 'public')}"
                )

# -----------------------------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="page-header">
        <div class="page-header-left">
            <div class="page-title">Addis Ababa Ride Demand Forecaster</div>
            <div class="page-subtitle">
                Operational dispatch &amp; demand planning platform powered by LightGBM ·
                <strong>Team Quatro</strong> (Led by Samuel Mitiku)
            </div>
            <span class="tag tag-blue">Competition Finalist</span>
            <span class="tag tag-green">RMSE 15.72 → 47.9% below baseline</span>
            <span class="tag tag-slate">12 Zones · Nov 2025</span>
            <span class="tag tag-amber">Zero-leakage chronological CV</span>
        </div>
        <div class="page-header-right">
            <strong>{selected_zone_display}</strong>
            {selected_date.strftime('%a, %b %d %Y')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Forecaster",
    "Model Evaluation",
    "EDA Insights",
    "About the Author",
])

# =============================================================================
# TAB 1 ─ LIVE OPERATIONAL FORECASTER
# =============================================================================
with tab1:

    # KPI row
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Peak Demand Hour</div>
                <div class="kpi-value">{peak_hour:02d}<span class="kpi-unit">:00</span></div>
                <div class="kpi-delta">{peak_trips:.0f} trips at peak</div>
            </div>
            """, unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card green">
                <div class="kpi-label">Recommended Fleet</div>
                <div class="kpi-value">{recommended_drivers:,}<span class="kpi-unit"> drivers</span></div>
                <div class="kpi-delta">{total_trips:,.0f} trips across 24 h</div>
            </div>
            """, unsafe_allow_html=True)
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card amber">
                <div class="kpi-label">Projected Revenue</div>
                <div class="kpi-value">{projected_revenue:,}<span class="kpi-unit"> ETB</span></div>
                <div class="kpi-delta">{fare_per_trip} ETB avg fare/trip</div>
            </div>
            """, unsafe_allow_html=True)
    with k4:
        weather_icon = "🌧" if total_rain_mm > 0 else "☀"
        weather_label = f"{total_rain_mm:.1f} mm" if total_rain_mm > 0 else "Clear"
        st.markdown(
            f"""
            <div class="kpi-card slate">
                <div class="kpi-label">Weather</div>
                <div class="kpi-value" style="font-size:1.5rem;">{weather_icon} {weather_label}</div>
                <div class="kpi-delta">{min_temp:.1f}°C – {max_temp:.1f}°C (EAT)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 24-Hour Demand Curve ──────────────────────────────────────────────────
    st.markdown(
        f"<div class='section-heading'>24-Hour Demand Forecast — {selected_zone_display}, "
        f"{selected_date.strftime('%B %d, %Y')}</div>",
        unsafe_allow_html=True,
    )

    chart_df = day_forecast.copy()

    # Confidence-style fill between 0 and effective_trips (area chart)
    area = (
        alt.Chart(chart_df)
        .mark_area(
            line={"color": ACCENT, "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=f"{ACCENT}55", offset=0),
                    alt.GradientStop(color=f"{ACCENT}08", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("hour:Q",
                    title="Hour of Day (EAT)",
                    scale=alt.Scale(domain=[0, 23]),
                    axis=alt.Axis(format="d", tickCount=12, grid=False,
                                  labelColor="#8892a4", titleColor="#8892a4")),
            y=alt.Y("effective_trips:Q",
                    title="Projected Trips",
                    axis=alt.Axis(grid=True, gridColor="#1c2333",
                                  labelColor="#8892a4", titleColor="#8892a4")),
            tooltip=[
                alt.Tooltip("hour_label:N", title="Time (EAT)"),
                alt.Tooltip("effective_trips:Q", title="Forecast Trips", format=".1f"),
                alt.Tooltip("temp_c:Q",          title="Temp (°C)",      format=".1f"),
                alt.Tooltip("rain_mm:Q",          title="Rain (mm)",      format=".2f"),
            ],
        )
    )

    dots = (
        alt.Chart(chart_df)
        .mark_circle(size=45, color=ACCENT_LIGHT, opacity=0.85)
        .encode(
            x="hour:Q",
            y="effective_trips:Q",
            tooltip=[
                alt.Tooltip("hour_label:N",       title="Time (EAT)"),
                alt.Tooltip("effective_trips:Q",   title="Forecast Trips", format=".1f"),
            ],
        )
    )

    # Peak annotation
    peak_df = pd.DataFrame([{
        "hour": peak_hour,
        "effective_trips": peak_trips,
        "label": f"Peak  {peak_trips:.0f} trips",
    }])
    peak_rule = (
        alt.Chart(peak_df)
        .mark_rule(color=DANGER, strokeDash=[3, 3], strokeWidth=1, opacity=0.8)
        .encode(x="hour:Q")
    )
    peak_dot = (
        alt.Chart(peak_df)
        .mark_point(color=DANGER, size=180, shape="triangle-up", filled=True)
        .encode(x="hour:Q", y="effective_trips:Q")
    )
    peak_text = (
        alt.Chart(peak_df)
        .mark_text(align="left", dx=9, dy=-10, color=DANGER,
                   fontWeight=600, fontSize=12, font="Inter")
        .encode(x="hour:Q", y="effective_trips:Q", text="label")
    )

    forecast_chart = (
        (area + dots + peak_rule + peak_dot + peak_text)
        .configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(forecast_chart, use_container_width=True)

    # ── Context bar ───────────────────────────────────────────────────────────
    rain_note  = (
        f"Rain expected — {total_rain_mm:.1f} mm total"
        if total_rain_mm > 0
        else "No precipitation forecast"
    )
    events_note = (
        "; ".join(day_events[:3])
        if day_events
        else "No major events scheduled in this zone"
    )
    st.markdown(
        f"""
        <div class="context-bar">
            <strong>Weather</strong>&ensp;{rain_note} &nbsp;·&nbsp;
            Temperature {min_temp:.1f}°C – {max_temp:.1f}°C &emsp;
            <strong>Events</strong>&ensp;{events_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dispatch table ────────────────────────────────────────────────────────
    with st.expander("View full 24-hour dispatch schedule"):
        table_df = pd.DataFrame({
            "Hour (EAT)":          day_forecast["hour_label"],
            "Forecast Trips":       day_forecast["effective_trips"].round(1),
            "Drivers":              (day_forecast["effective_trips"] / trips_per_driver).round(1),
            "Revenue (ETB)":        (day_forecast["effective_trips"] * fare_per_trip).round(0).astype(int),
            "Temp (°C)":            day_forecast["temp_c"].round(1),
            "Rain (mm)":            day_forecast["rain_mm"].round(2),
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download dispatch plan (.csv)",
            data=csv_bytes,
            file_name=f"dispatch_{selected_db_zone}_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 2 ─ MODEL EVALUATION
# =============================================================================
with tab2:

    # Summary stats row
    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-number">15.72</div>
                <div class="stat-label">Final RMSE</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">6.78</div>
                <div class="stat-label">Final MAE</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">47.9%</div>
                <div class="stat-label">RMSE vs. Mean Baseline</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">2.28 s</div>
                <div class="stat-label">Training Time</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">28</div>
                <div class="stat-label">Final Feature Count</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Scoreboard + Chart ────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 2])

    leaderboard_df = pd.DataFrame([
        {"Model":           "LightGBM",
         "RMSE": 15.72, "MAE": 6.78, "Fit Time (s)": 2.28,
         "Champion": True},
        {"Model":           "Random Forest",
         "RMSE": 15.78, "MAE": 6.58, "Fit Time (s)": 25.24,
         "Champion": False},
        {"Model":           "HistGradient Boosting",
         "RMSE": 15.78, "MAE": 6.88, "Fit Time (s)": 3.87,
         "Champion": False},
        {"Model":           "Seasonal Naive (baseline)",
         "RMSE": 17.16, "MAE": 7.53, "Fit Time (s)": 0.01,
         "Champion": False},
        {"Model":           "Ridge Regression",
         "RMSE": 25.44, "MAE": 16.87, "Fit Time (s)": 0.21,
         "Champion": False},
        {"Model":           "Global Mean",
         "RMSE": 30.16, "MAE": 20.79, "Fit Time (s)": 0.01,
         "Champion": False},
    ])

    with sc1:
        st.markdown("<div class='section-heading'>Holdout Scoreboard — October 2025 (8,690 zone-hours)</div>", unsafe_allow_html=True)
        display_df = leaderboard_df.drop(columns=["Champion"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(
            "LightGBM selected as final model: best RMSE, 10× faster than Random Forest, "
            "with native categorical zone handling."
        )

    with sc2:
        st.markdown("<div class='section-heading'>RMSE Comparison</div>", unsafe_allow_html=True)
        bar_chart = (
            alt.Chart(leaderboard_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("RMSE:Q", title="Validation RMSE",
                         axis=alt.Axis(grid=True, gridColor="#1c2333",
                                       labelColor="#8892a4", titleColor="#8892a4")),
                y=alt.Y("Model:N", sort="-x", title=None,
                         axis=alt.Axis(labelColor="#8892a4")),
                color=alt.condition(
                    alt.datum["Champion"],
                    alt.value(ACCENT),
                    alt.value(SLATE),
                ),
                tooltip=["Model", "RMSE", "MAE", "Fit Time (s)"],
            )
            .configure(background="transparent")
            .configure_view(strokeOpacity=0)
            .properties(height=230)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── CV + Ablation ─────────────────────────────────────────────────────────
    cv1, cv2 = st.columns(2)

    with cv1:
        st.markdown("<div class='section-heading'>Rolling-Origin Cross-Validation (4 × 14-day folds)</div>", unsafe_allow_html=True)
        cv_df = pd.DataFrame([
            {"Fold": 1, "Window": "Aug 18–31", "Train Rows": 62057,
             "LightGBM RMSE": 10.97, "Naive RMSE": 12.84},
            {"Fold": 2, "Window": "Sep 01–14", "Train Rows": 65991,
             "LightGBM RMSE": 15.00, "Naive RMSE": 17.97},
            {"Fold": 3, "Window": "Sep 15–28", "Train Rows": 69925,
             "LightGBM RMSE": 12.94, "Naive RMSE": 15.22},
            {"Fold": 4, "Window": "Sep 29–Oct 12", "Train Rows": 73848,
             "LightGBM RMSE": 17.08, "Naive RMSE": 18.04},
        ])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.success("LightGBM won every fold · Aggregate **13.998 ± 2.633 RMSE**")

    with cv2:
        st.markdown("<div class='section-heading'>Feature Group Ablation</div>", unsafe_allow_html=True)
        abl_df = pd.DataFrame([
            {"Feature Group": "Calendar + Zone + Trend (base)",
             "n Features": 8,  "RMSE": 16.92, "Δ RMSE": "—"},
            {"Feature Group": "+ Weather (temp, rain, humidity)",
             "n Features": 15, "RMSE": 15.99, "Δ RMSE": "−0.93"},
            {"Feature Group": "+ Event windows (lead/lag)",
             "n Features": 21, "RMSE": 16.72, "Δ RMSE": "−0.20"},
            {"Feature Group": "+ Combined (weather + events)",
             "n Features": 28, "RMSE": 15.72, "Δ RMSE": "−1.20"},
        ])
        st.dataframe(abl_df, use_container_width=True, hide_index=True)
        st.info("Weather integration delivers the largest single lift (−0.93 RMSE).")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Leakage audit ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Production Leakage Audit</div>", unsafe_allow_html=True)
    la1, la2 = st.columns([2, 1])
    with la1:
        st.markdown(
            """
            In live dispatch, fields like `avg_fare_birr`, `avg_wait_min`, and `active_drivers`
            are **outcomes of demand**, not predictors. Including them artificially inflated
            holdout RMSE by **19.9%** (13.55 vs 16.92). All three were excluded from all
            feature tables after a full audit, ensuring the model is production-safe.
            """
        )
    with la2:
        leakage_table = pd.DataFrame([
            {"Feature":         "avg_fare_birr",  "Available Pre-Dispatch": "No"},
            {"Feature":         "avg_wait_min",   "Available Pre-Dispatch": "No"},
            {"Feature":         "active_drivers", "Available Pre-Dispatch": "No"},
            {"Feature":         "hour",           "Available Pre-Dispatch": "Yes"},
            {"Feature":         "temp_c (forecast)", "Available Pre-Dispatch": "Yes"},
            {"Feature":         "event_any",      "Available Pre-Dispatch": "Yes"},
        ])
        st.dataframe(leakage_table, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3 ─ EDA INSIGHTS
# =============================================================================
with tab3:

    st.markdown(
        "<div class='section-heading'>Key Discoveries That Shaped Feature Engineering</div>",
        unsafe_allow_html=True,
    )

    def eda_card(col, fig_name: str, number: str, title: str, takeaway: str):
        fig_path = FIGURES_DIR / fig_name
        with col:
            st.markdown(f"**{number} — {title}**")
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            st.caption(takeaway)

    r1c1, r1c2 = st.columns(2)
    eda_card(
        r1c1,
        "fig06_weather_timezone_check.png",
        "01", "UTC → EAT Timezone Correction",
        "Raw weather peaked at 11:00 UTC. Empirical solar alignment proved the sensor "
        "stream was UTC — a +3 h shift to EAT was mandatory for model accuracy.",
    )
    eda_card(
        r1c2,
        "fig07_rain_effect.png",
        "02", "Non-Linear Rain Surge",
        "Light-to-moderate rain drove a 15–30% demand lift as commuters sought shelter, "
        "while heavy rainfall caused saturation and cancellations — a key non-linearity.",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    eda_card(
        r2c1,
        "fig08_event_study.png",
        "03", "Asymmetric Event Surge Windows",
        "Pre-event arrivals build gradually over 3 h; post-event departures create a sharp "
        "1.55× surge compressed into 2 h — motivating separate lead and lag feature columns.",
    )
    eda_card(
        r2c2,
        "fig04_hour_by_weekday_heatmap.png",
        "04", "Commercial vs. Residential Diurnal Patterns",
        "Merkato and Kazanchis show clear dual commute peaks (08:00 & 18:00), while Bole "
        "sustains strong late-night weekend demand from airport and hospitality traffic.",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    eda_card(
        r3c1,
        "fig03_demand_trend_with_holidays.png",
        "05", "Holiday Depression Effects",
        "Major religious holidays (Timkat, Meskel, Genna) depressed daily trips by 20–35%, "
        "requiring explicit negative seasonal adjustments in the calendar feature set.",
    )
    eda_card(
        r3c2,
        "fig12_feature_importance.png",
        "06", "LightGBM Feature Importance",
        "Hour of day, zone identity, and weekday type dominate the importance ranking, "
        "with temperature and event-active windows contributing the primary external lift.",
    )


# =============================================================================
# TAB 4 ─ ABOUT THE AUTHOR
# =============================================================================
with tab4:

    a1, a2 = st.columns([2, 3])

    with a1:
        st.markdown(
            """
            <div class="profile-card">
                <div class="profile-name">Samuel Mitiku</div>
                <div class="profile-title">Project Lead · ML Systems Architect · Team Quatro</div>
                <div class="profile-bio">
                    Led an interdisciplinary team through the full data science lifecycle —
                    from raw sensor ingestion and UTC timezone resolution to LightGBM
                    hyperparameter search and live Streamlit deployment. Specialises in
                    production-safe time-series modeling with strict leakage prevention and
                    interpretable operational dashboards.
                </div>
                <div class="profile-skills-title">Demonstrated Competencies</div>
                <div class="profile-skill-item">→ End-to-end Time-Series Forecasting</div>
                <div class="profile-skill-item">→ LightGBM Optimization &amp; Feature Ablation</div>
                <div class="profile-skill-item">→ Leakage-Proof Rolling-Origin CV Design</div>
                <div class="profile-skill-item">→ Multi-Source Data Integration (Weather + Events)</div>
                <div class="profile-skill-item">→ Operational ML Dashboards (Streamlit)</div>
                <br>
                <a href="https://github.com/samuelmitiku393/team_quatro" target="_blank" class="link-btn link-btn-blue">
                    ↗ GitHub Project
                </a>
                <a href="https://linkedin.com" target="_blank" class="link-btn link-btn-green">
                    ↗ LinkedIn Profile
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown("<div class='section-heading'>System Architecture</div>", unsafe_allow_html=True)
        st.markdown(
            """
            ```
            ┌─────────────────────────────────────────────────────────────────┐
            │  Data Sources                                                    │
            │  ├─ Trip Logs (12 zones, Jan–Oct 2025, ~83k zone-hours)         │
            │  ├─ Hourly Weather (temp, rain, humidity, wind) — UTC sensor     │
            │  └─ City Events Calendar (football, concerts, public holidays)   │
            └───────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────────────────────┐
            │  Data Engineering Layer                                          │
            │  ├─ Timestamp normalization & deduplication (cleaning.py)        │
            │  ├─ UTC → EAT (+3 h) conversion & weather join (integration.py)  │
            │  └─ Asymmetric event lead/lag windows & leakage audit            │
            └───────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────────────────────┐
            │  Feature Store (28 features)                                     │
            │  ├─ Diurnal: hour, dayofweek, is_weekend, weekofyear            │
            │  ├─ Weather: temp_c, rain_mm, rain_3h, rain_class, humidity     │
            │  └─ Events: event_any, event_active, hours_until/since, type    │
            └───────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────────────────────┐
            │  Model Training (train.py)                                       │
            │  ├─ Rolling-origin CV, 4 × 14-day folds, zero future leakage    │
            │  ├─ Ablation: calendar → +weather → +events → combined          │
            │  └─ RandomizedSearchCV, 15 trials, TimeSeriesSplit(3)           │
            └───────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────────────────────┐
            │  LightGBM Regressor  (RMSE 15.72, MAE 6.78)                     │
            │  num_leaves=127  ·  lr=0.02  ·  colsample=0.9  ·  subsample=1  │
            └───────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────────────────────┐
            │  Streamlit Dashboard (app.py)                                    │
            │  ├─ Zone + date selector → 24-hour demand curve                 │
            │  ├─ Fleet sizing & revenue projection (adjustable parameters)    │
            │  └─ Scenario stress testing, weather & events context lookup    │
            └─────────────────────────────────────────────────────────────────┘
            ```
            """
        )

        st.markdown("<div class='section-heading'>Technology Stack</div>", unsafe_allow_html=True)
        tech_c1, tech_c2 = st.columns(2)
        with tech_c1:
            st.markdown(
                """
                **Modeling**
                - LightGBM, Scikit-Learn
                - Joblib (model serialisation)
                - Pandas, NumPy

                **Validation**
                - Rolling-origin time-series CV
                - RandomizedSearchCV
                - Chronological train/holdout splits
                """
            )
        with tech_c2:
            st.markdown(
                """
                **Interface & Visualisation**
                - Streamlit (dashboard framework)
                - Altair (declarative charts)
                - Matplotlib (static figures)

                **Deployment**
                - Streamlit Community Cloud
                - GitHub Actions CI/CD
                - Docker-compatible
                """
            )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="page-footer">
        Addis Ababa Ride Demand Forecaster &nbsp;·&nbsp;
        <strong>Team Quatro</strong> (Led by Samuel Mitiku) &nbsp;·&nbsp;
        Powered by LightGBM + Streamlit &nbsp;·&nbsp;
        <a href="https://github.com/samuelmitiku393/team_quatro">View source on GitHub ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
