# frontend/app.py
# Run:
# streamlit run frontend/app.py

import streamlit as st
import requests
from datetime import date
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# CONFIG
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# Local mode
# API_URL = "http://localhost:8000/predict"
# PROM_URL = "http://localhost:9090"

# Docker mode:
API_URL = "http://api:8000/predict"
RECENT_URL = "http://api:8000/recent_predictions"
PROM_URL = "http://prometheus:9090"
GRAFANA_URL = "http://grafana:3000"

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --border:    #1f2d45;
    --accent:    #3b82f6;
    --accent-2:  #06b6d4;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --radius:    12px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: var(--muted) !important;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text) !important;
}

/* ── Page heading ── */
.page-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}
.page-subtitle {
    font-size: 0.875rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.2rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card.success::before { background: var(--success); }
.metric-card.warning::before { background: var(--warning); }
.metric-card.danger::before  { background: var(--danger);  }
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div > input {
    background: #1a2236 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stDateInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Alerts ── */
.stAlert {
    background: #1a2236 !important;
    border-radius: 8px !important;
    border-left-width: 3px !important;
}

/* ── Streamlit native metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; color: var(--text) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Sidebar badge ── */
.sidebar-badge {
    display: inline-block;
    background: #1a2236;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}

/* ── Status dot ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--muted);
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.live    { background: var(--success); box-shadow: 0 0 6px var(--success); animation: pulse 2s infinite; }
.dot.offline { background: var(--danger); }
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── Manual section ── */
.manual-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin: 0.8rem 0;
    padding: 0.9rem 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--accent);
    background: #1a2236;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    white-space: nowrap;
    margin-top: 2px;
}
.step-text {
    font-size: 0.875rem;
    color: var(--text);
    line-height: 1.5;
}

/* ── Port table ── */
.port-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.port-row:last-child { border-bottom: none; }
.port-name { color: var(--muted); }
.port-val  { font-family: 'DM Mono', monospace; color: var(--accent-2); font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS 
def prom_query(query):
    try:
        url = f"{PROM_URL}/api/v1/query"
        r = requests.get(url, params={"query": query}, timeout=5)
        data = r.json()
        if data["status"] == "success" and len(data["data"]["result"]) > 0:
            return float(data["data"]["result"][0]["value"][1])
        return None
    except:
        return None


# ── SIDEBAR 
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                    background:linear-gradient(135deg,#60a5fa,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DEMAND FORECAST
        </div>
        <div style="font-size:0.7rem;color:#64748b;letter-spacing:1px;margin-top:2px;">
            ML · RETAIL · ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Forecast Dashboard", "Monitoring Dashboard", "User Manual"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-badge">v1.0 · Production</div>',
        unsafe_allow_html=True
    )


# ── PAGE 1: FORECAST 
if page == "Forecast Dashboard":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Demand Forecast</div>
        <div class="page-subtitle">Predict product demand using deployed ML model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Forecast Parameters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        store = st.selectbox("🏬 Store", list(range(1, 11)))
    with c2:
        item = st.selectbox("📦 Item", list(range(1, 51)))
    with c3:
        pred_date = st.date_input("📅 Date", date.today())

    st.markdown("</div>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run = st.button("Run Forecast →", use_container_width=True)

    if run:
        payload = {
            "store": int(store),
            "item": int(item),
            "date": str(pred_date)
        }
        try:
            with st.spinner("Generating forecast..."):
                r = requests.post(API_URL, json=payload, timeout=10)

            if r.status_code == 200:
                data = r.json()

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card success">
                        <div class="metric-label">Predicted Sales</div>
                        <div class="metric-value">{data["predicted_sales"]}</div>
                        <div class="metric-unit">units · store {store} · item {item}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Inference Latency</div>
                        <div class="metric-value">{data["latency_ms"]}</div>
                        <div class="metric-unit">milliseconds</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success("Prediction completed successfully.")
            else:
                st.error(r.text)

        except Exception as e:
            st.error(f"Backend unavailable: {e}")

    st.markdown("---")

    st.markdown("""
    <div class="card">
        <div class="card-title">Live Recent Predictions</div>
    """, unsafe_allow_html=True)

    # refresh every 5 sec
    st_autorefresh(interval=5000, key="live_preds")

    try:
        r = requests.get(RECENT_URL, timeout=5)

        if r.status_code == 200:
            df = pd.DataFrame(r.json())

            if not df.empty:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No predictions available yet.")

        else:
            st.warning("Could not load recent predictions.")

    except Exception:
        st.warning("Recent predictions service unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

# ── PAGE 2: MONITORING ────────────────────────────────────────────────────────
elif page == "Monitoring Dashboard":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Monitoring</div>
        <div class="page-subtitle">Live metrics pulled from Prometheus</div>
    </div>
    """, unsafe_allow_html=True)

    req_count    = prom_query("api_requests_total")
    err_count    = prom_query("api_errors_total")
    latency_sum   = prom_query("api_latency_seconds_sum")
    latency_count = prom_query("api_latency_seconds_count")

    avg_latency_ms = None
    if latency_sum is not None and latency_count not in (None, 0):
        avg_latency_ms = (latency_sum / latency_count) * 1000

    live = req_count is not None
    dot_cls = "live" if live else "offline"
    dot_label = "Connected" if live else "Unreachable"

    st.markdown(f"""
    <div class="status-row" style="margin-bottom:1.5rem;">
        <span class="dot {dot_cls}"></span>
        <span>Prometheus · {dot_label}</span>
    </div>
    """, unsafe_allow_html=True)

    req_val = int(req_count) if req_count is not None else "—"
    err_val = int(err_count) if err_count is not None else "—"
    lat_val = f"{round(avg_latency_ms, 2)}" if avg_latency_ms is not None else "—"

    err_cls = "danger" if err_count and err_count > 0 else ""

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total Requests</div>
            <div class="metric-value">{req_val}</div>
            <div class="metric-unit">all time</div>
        </div>
        <div class="metric-card {err_cls}">
            <div class="metric-label">Total Errors</div>
            <div class="metric-value">{err_val}</div>
            <div class="metric-unit">failures</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value">{lat_val}</div>
            <div class="metric-unit">milliseconds</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="card">
        <div class="card-title">Monitoring Endpoints</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="port-row">
        <span class="port-name">Prometheus UI</span>
        <span class="port-val">{PROM_URL}</span>
    </div>
    <div class="port-row">
        <span class="port-name">FastAPI Metrics</span>
        <span class="port-val">http://api:8000/metrics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not live:
        st.warning(
            "Prometheus not reachable. Start the Prometheus container "
            "or verify scrape configuration."
        )


# ── PAGE 3: USER MANUAL ───────────────────────────────────────────────────────
elif page == "User Manual":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">User Manual</div>
        <div class="page-subtitle">How to use the Retail Demand Forecasting application</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">Generating a Forecast</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Open the <strong>Forecast Dashboard</strong> from the sidebar."),
        ("02", "Select a <strong>Store ID</strong> (1–10) from the dropdown."),
        ("03", "Select an <strong>Item ID</strong> (1–50) from the dropdown."),
        ("04", "Choose the target <strong>Date</strong> for the prediction."),
        ("05", "Click <strong>Run Forecast →</strong> to call the model API."),
        ("06", "View the <strong>Predicted Sales</strong> and <strong>Latency</strong> results."),
    ]

    for num, text in steps:
        st.markdown(f"""
        <div class="manual-step">
            <div class="step-num">{num}</div>
            <div class="step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="card-title" style="margin-top:1rem;">Service Ports</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="port-row">
            <span class="port-name">Frontend (Streamlit)</span>
            <span class="port-val">:8501</span>
        </div>
        <div class="port-row">
            <span class="port-name">API (FastAPI)</span>
            <span class="port-val">:8000</span>
        </div>
        <div class="port-row">
            <span class="port-name">Prometheus</span>
            <span class="port-val">:9090</span>
        </div>
        <div class="port-row">
            <span class="port-name">Grafana</span>
            <span class="port-val">:3000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="card-title">Intended Users</div>', unsafe_allow_html=True)
    for role in ["Store Managers", "Supply Chain Teams", "Demand Planners", "Business Analysts"]:
        st.markdown(f"""
        <div style="display:inline-block;margin:0.25rem 0.25rem 0.25rem 0;
                    background:#1a2236;border:1px solid #1f2d45;border-radius:999px;
                    padding:0.25rem 0.9rem;font-size:0.8rem;color:#94a3b8;">
            {role}
        </div>
        """, unsafe_allow_html=True)