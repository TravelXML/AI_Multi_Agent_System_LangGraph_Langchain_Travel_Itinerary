import os
import urllib.parse
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from langchain_core.messages import HumanMessage
import main
import mcp_client
from main import app
from export_utils import render_pdf, render_docx

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-travel-planner")


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_langsmith_stats():
    """Pull aggregate + per-agent LLM metrics from LangSmith for the observability panel."""
    from langsmith import Client

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    proj = client.read_project(project_name=LANGSMITH_PROJECT, include_stats=True)

    if not proj.run_count:
        return None

    by_agent = {}
    recent_runs = []
    for run in client.list_runs(project_name=LANGSMITH_PROJECT, run_type="llm", limit=100):
        node = ((run.extra or {}).get("metadata", {}) or {}).get("langgraph_node", "other")
        entry = by_agent.setdefault(
            node, {"calls": 0, "tokens": 0, "latency": 0.0, "errors": 0}
        )
        is_error = bool(run.error) or (run.status not in (None, "success"))
        entry["calls"] += 1
        entry["tokens"] += run.total_tokens or 0
        if run.latency is not None:
            entry["latency"] += run.latency
        if is_error:
            entry["errors"] += 1
        recent_runs.append({
            "time": run.start_time, "agent": node, "status": run.status,
            "tokens": run.total_tokens or 0, "latency": run.latency or 0,
        })

    return {
        "run_count": proj.run_count,
        "total_tokens": proj.total_tokens or 0,
        "prompt_tokens": proj.prompt_tokens or 0,
        "completion_tokens": proj.completion_tokens or 0,
        "latency_p50": proj.latency_p50.total_seconds() if proj.latency_p50 else 0,
        "error_rate": proj.error_rate or 0,
        "total_cost": proj.total_cost,
        "project_url": proj.url,
        "by_agent": by_agent,
        "recent_runs": recent_runs,
    }


def render_observability_panel():
    st.markdown("<div class='sidebar-title'>Observability</div>", unsafe_allow_html=True)

    if not os.getenv("LANGCHAIN_API_KEY"):
        st.markdown("<div class='sidebar-chip'>⚪ LangSmith not configured</div>", unsafe_allow_html=True)
        st.caption("Add `LANGCHAIN_API_KEY` to `.env` to see live tracing metrics here.")
        return

    if st.button("🔄 Refresh metrics", key="obs_refresh", use_container_width=True):
        _fetch_langsmith_stats.clear()

    try:
        stats = _fetch_langsmith_stats()
    except Exception as e:
        st.caption(f"⚠️ Couldn't reach LangSmith: {e}")
        return

    if stats is None:
        st.caption("No traces yet — generate a trip plan to see metrics.")
        return

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Traced Runs", stats["run_count"])
        st.metric("Median Latency", f"{stats['latency_p50']:.1f}s")
    with m2:
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
        st.metric("Success Rate", f"{100 - stats['error_rate'] * 100:.0f}%")

    st.caption(f"↳ {stats['prompt_tokens']:,} prompt · {stats['completion_tokens']:,} completion tokens")

    if stats["by_agent"]:
        st.markdown(
            "<div class='sidebar-title' style='margin-top:0.6rem;font-size:0.78rem;'>By Agent</div>",
            unsafe_allow_html=True,
        )
        for node, info in sorted(stats["by_agent"].items(), key=lambda kv: -kv[1]["tokens"]):
            calls = info["calls"]
            avg_latency = info["latency"] / calls if calls else 0
            success_rate = (calls - info["errors"]) / calls * 100 if calls else 0
            st.markdown(
                f"<div class='sidebar-chip'>{node} — {calls} call(s) · "
                f"{info['tokens']:,} tok · {avg_latency:.1f}s avg · {success_rate:.0f}% ok</div>",
                unsafe_allow_html=True,
            )

    _recent_count = len(main.get_recent_searches(20))
    with st.container(key="nav_obs"):
        if st.button(f"🔍 Internal Observability ({_recent_count} recent searches)",
                     key="nav_observability", use_container_width=True):
            st.session_state.view = "observability"
            st.rerun()

    st.link_button("📊 Open Full Dashboard ↗", stats["project_url"], use_container_width=True)


def render_observability_page():
    """Full-page version of Internal Observability — recent searches + per-agent
    latency/success/error charts. Navigated to from the sidebar (not an inline expander)."""
    st.markdown(
        "<div class='top-panel'>🔍 <b>Internal Observability</b>"
        "<span class='top-panel-sep'>—</span>"
        "<span>Recent searches, per-agent latency, tokens, success &amp; error rates.</span></div>",
        unsafe_allow_html=True,
    )

    if st.button("← Back to Trip Planner", key="back_home"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown("<div class='sec-head'><span>🕓 Recent Searches</span></div>", unsafe_allow_html=True)
    _recent = main.get_recent_searches(20)
    if not _recent:
        st.caption("No searches yet — generate a trip plan to start building history.")
    else:
        df = pd.DataFrame(_recent)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%d %b, %H:%M")
        df["dates"] = df.apply(
            lambda r: f"{r['start_date']} → {r['end_date']}" if r["start_date"] else "—", axis=1
        )
        df["query"] = df["query"].fillna("").str.slice(0, 70)
        df["response"] = df["response"].fillna("").str.slice(0, 90) + "…"
        st.dataframe(
            df[["created_at", "user_id", "model", "tokens_used", "dates", "query", "response"]]
              .rename(columns={
                  "created_at": "When", "user_id": "User", "model": "Model",
                  "tokens_used": "Tokens", "dates": "Trip Dates",
                  "query": "Query", "response": "Response",
              }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("<div class='sec-head'><span>📊 Agent Performance</span></div>", unsafe_allow_html=True)
    if not os.getenv("LANGCHAIN_API_KEY"):
        st.caption("Add `LANGCHAIN_API_KEY` to `.env` to enable this section.")
        return

    try:
        stats = _fetch_langsmith_stats()
    except Exception as e:
        st.caption(f"⚠️ Couldn't reach LangSmith: {e}")
        return

    if not (stats and stats["by_agent"]):
        st.caption("No traced agent calls yet — generate a trip plan to populate this dashboard.")
        return

    agent_rows = []
    for node, info in stats["by_agent"].items():
        calls = info["calls"]
        agent_rows.append({
            "Agent": node,
            "Avg Latency (s)": round(info["latency"] / calls, 1) if calls else 0,
            "Success Rate (%)": round((calls - info["errors"]) / calls * 100, 0) if calls else 0,
            "Error Rate (%)": round(info["errors"] / calls * 100, 0) if calls else 0,
        })
    agent_df = pd.DataFrame(agent_rows).set_index("Agent")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Latency by Agent**")
        st.bar_chart(agent_df[["Avg Latency (s)"]], color="#06b6d4")
    with c2:
        st.markdown("**Success Rate by Agent**")
        st.bar_chart(agent_df[["Success Rate (%)"]], color="#22c55e")
    with c3:
        st.markdown("**Error Rate by Agent**")
        st.bar_chart(agent_df[["Error Rate (%)"]], color="#ef4444")

    st.markdown("**By Agent — Detail**")
    rows = []
    for node, info in sorted(stats["by_agent"].items(), key=lambda kv: -kv[1]["tokens"]):
        calls = info["calls"]
        success_rate = (calls - info["errors"]) / calls * 100 if calls else 0
        error_rate = info["errors"] / calls * 100 if calls else 0
        avg_latency = info["latency"] / calls if calls else 0
        avg_tokens = info["tokens"] / calls if calls else 0
        rows.append({
            "Agent": node, "Calls": calls,
            "Success Rate": f"{success_rate:.0f}%", "Error Rate": f"{error_rate:.0f}%",
            "Avg Latency": f"{avg_latency:.1f}s", "Total Tokens": f"{info['tokens']:,}",
            "Avg Tokens/Call": f"{avg_tokens:.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    runs = stats.get("recent_runs") or []
    if runs:
        st.markdown(f"**Individual LLM Calls** (last {len(runs)})")
        rdf = pd.DataFrame(runs).sort_values("time", ascending=False)
        rdf["time"] = pd.to_datetime(rdf["time"]).dt.strftime("%d %b, %H:%M:%S")
        rdf["latency"] = rdf["latency"].map(lambda x: f"{x:.1f}s")
        rdf["tokens"] = rdf["tokens"].map(lambda x: f"{x:,}")
        st.dataframe(
            rdf.rename(columns={
                "time": "When", "agent": "Agent", "status": "Status",
                "tokens": "Tokens", "latency": "Latency",
            }),
            use_container_width=True, hide_index=True,
        )

    st.link_button("📊 Open Full LangSmith Dashboard ↗", stats["project_url"], use_container_width=True)

MODEL_OPTIONS = {
    "Nemotron 3 Super 120B — fast, default": "nvidia/nemotron-3-super-120b-a12b:free",
    "Nemotron 3 Ultra 550B — most powerful": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "GPT-OSS 20B": "openai/gpt-oss-20b:free",
    "Gemma 4 31B": "google/gemma-4-31b-it:free",
    "GLM 5.2": "z-ai/glm-5.2:free",
}

CITIES = [
    "Delhi, India", "Mumbai, India", "Bengaluru, India", "Chennai, India", "Kolkata, India",
    "Hyderabad, India", "Pune, India", "Ahmedabad, India", "Jaipur, India", "Goa, India",
    "Kochi, India", "Chandigarh, India", "Lucknow, India", "Amritsar, India", "Varanasi, India",
    "Udaipur, India", "Shimla, India", "Manali, India", "Rishikesh, India", "Darjeeling, India",
    "Tokyo, Japan", "Osaka, Japan", "Kyoto, Japan", "Seoul, South Korea", "Busan, South Korea",
    "Bangkok, Thailand", "Phuket, Thailand", "Chiang Mai, Thailand", "Singapore, Singapore",
    "Kuala Lumpur, Malaysia", "Langkawi, Malaysia", "Bali, Indonesia", "Jakarta, Indonesia",
    "Hanoi, Vietnam", "Ho Chi Minh City, Vietnam", "Da Nang, Vietnam", "Manila, Philippines",
    "Cebu, Philippines", "Hong Kong, Hong Kong", "Beijing, China", "Shanghai, China",
    "Taipei, Taiwan", "Dubai, UAE", "Abu Dhabi, UAE", "Doha, Qatar", "Muscat, Oman",
    "Riyadh, Saudi Arabia", "Istanbul, Turkey", "Antalya, Turkey", "Tel Aviv, Israel",
    "Cairo, Egypt", "Amman, Jordan", "Paris, France", "Nice, France", "London, United Kingdom",
    "Edinburgh, United Kingdom", "Rome, Italy", "Milan, Italy", "Venice, Italy", "Florence, Italy",
    "Barcelona, Spain", "Madrid, Spain", "Ibiza, Spain", "Amsterdam, Netherlands",
    "Berlin, Germany", "Munich, Germany", "Frankfurt, Germany", "Zurich, Switzerland",
    "Interlaken, Switzerland", "Vienna, Austria", "Prague, Czech Republic", "Athens, Greece",
    "Santorini, Greece", "Mykonos, Greece", "Lisbon, Portugal", "Porto, Portugal",
    "Dublin, Ireland", "Copenhagen, Denmark", "Stockholm, Sweden", "Oslo, Norway",
    "Reykjavik, Iceland", "Moscow, Russia", "Budapest, Hungary", "Warsaw, Poland",
    "Brussels, Belgium", "New York, USA", "Los Angeles, USA", "San Francisco, USA",
    "Las Vegas, USA", "Miami, USA", "Chicago, USA", "Orlando, USA", "Honolulu, USA",
    "Toronto, Canada", "Vancouver, Canada", "Mexico City, Mexico", "Cancun, Mexico",
    "Rio de Janeiro, Brazil", "Buenos Aires, Argentina", "Lima, Peru", "Santiago, Chile",
    "Sydney, Australia", "Melbourne, Australia", "Gold Coast, Australia",
    "Auckland, New Zealand", "Queenstown, New Zealand", "Cape Town, South Africa",
    "Marrakech, Morocco", "Nairobi, Kenya", "Mauritius, Mauritius", "Seychelles, Seychelles",
    "Maldives, Maldives", "Colombo, Sri Lanka", "Kathmandu, Nepal", "Thimphu, Bhutan",
]

TRAVEL_TYPES = ["Leisure", "Family", "Honeymoon", "Adventure", "Business", "Backpacking", "Luxury", "Solo"]

st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #080d14;
}

/* ── Compact top panel — full-bleed cyan banner, flush with top ── */
[data-testid="stMain"] .block-container { padding-top: 0 !important; }
.top-panel {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.88rem;
    color: #ffffff;
    padding: 0.85rem 5rem;
    margin: 0 -5rem 1.1rem -5rem;
    background: #06b6d4;
    border: none;
    border-radius: 0;
    height: 3rem;
    box-sizing: border-box;
}
.top-panel b {
    color: #ffffff;
    font-size: 1.05rem;
    white-space: nowrap;
}
.top-panel span { color: #ffffff; }
.top-panel-sep { color: rgba(255,255,255,0.65); }

/* ── Input card ── */
.input-card {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.input-label {
    color: #67e8f9;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Quick destinations ── */
.dest-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.2rem;
}
.dest-chip {
    background: #111b2b;
    border: 1px solid #1e3050;
    color: #f7fdf4;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
}
.dest-chip:hover { background: #1a2e47; border-color: #06b6d4; color: #fff; }

/* ── Generate button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 50%, #164e63 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    box-shadow: 0 0 24px rgba(6,182,212,0.35), 0 4px 15px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 40px rgba(6,182,212,0.6), 0 6px 20px rgba(0,0,0,0.5) !important;
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #22d3ee 0%, #0891b2 50%, #0e7490 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Agent status cards ── */
[data-testid="stStatusWidget"] {
    background: #0e1a2e !important;
    border: 1px solid #1e3050 !important;
    border-radius: 12px !important;
}
[data-testid="stStatusWidget"] > div:first-child {
    background: #0e1a2e !important;
    border-radius: 12px 12px 0 0 !important;
}
[data-testid="stStatusWidget"] details,
[data-testid="stStatusWidget"] details > div,
[data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] {
    background: #0a1520 !important;
    color: #ffffff !important;
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stStatusWidget"] * { color: #ffffff !important; }
[data-testid="stStatusWidget"] a { color: #67e8f9 !important; }
[data-testid="stStatusWidget"] hr { border-color: #1e3050 !important; }

/* ── Section headers ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2e44;
}
.sec-head span { font-size: 1.15rem; font-weight: 600; color: #e0edf8; }

/* ── Metric bar ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-box {
    flex: 1;
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #67e8f9; }
.metric-lbl { font-size: 0.78rem; color: #5a7a96; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Final plan ── */
.final-card {
    background: linear-gradient(160deg, #0c1a2e 0%, #0a1520 100%);
    border: 1px solid #1e3a5c;
    border-left: 4px solid #06b6d4;
    border-radius: 14px;
    padding: 1.8rem;
    line-height: 1.8;
    color: #cce0f5;
    font-size: 0.95rem;
}

/* ── Save bar ── */
.save-bar {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    color: #5a8ab0;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090e18 !important;
    border-right: 1px solid #141f30 !important;
}
.sidebar-chip {
    background: #0e1a2b;
    border: 1px solid #1a2e44;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
    color: #7aa8cc;
}
.sidebar-title { color: #e0edf8; font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; }

/* Hide branding — but keep the toolbar's sidebar-expand button working.
   toolbarMode="minimal" (config.toml) already strips the Deploy/GitHub buttons. */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* Textarea */
.stTextArea textarea {
    background: #0a1520 !important;
    border: 1px solid #1e2e44 !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
}
.stTextArea textarea::placeholder { color: #4a6a85 !important; }

/* Text input (sidebar User ID field) */
input[type="text"], .stTextInput input {
    background: #0e1a2b !important;
    border: 1px solid #1a2e44 !important;
    border-radius: 8px !important;
    color: #e0edf8 !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
}
input[type="text"]::placeholder { color: #3a5570 !important; }

/* All Streamlit labels — dark bg → light text */
.stTextInput label, .stTextArea label,
.stSelectbox label, .stNumberInput label {
    color: #67e8f9 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
}

/* General markdown / paragraph text */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: #cce0f5 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e8f4ff !important; }
.stMarkdown code {
    background: #0e1a2b !important;
    color: #67e8f9 !important;
    padding: 0.15em 0.4em;
    border-radius: 4px;
}

/* Metric labels — was #5a7a96 (too dim on dark bg) */
.metric-lbl { color: #7aa8cc !important; }

/* Save bar — was #5a8ab0 (slightly dim) */
.save-bar { color: #8ab8d8 !important; }
.save-bar code { color: #67e8f9 !important; background: #0a1520 !important; }

/* Streamlit warning / info / success on dark bg */
.stAlert { background: #0e1a2b !important; border-radius: 10px !important; }
.stAlert p, .stAlert div { color: #e0edf8 !important; }

/* Sidebar text & dividers */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown { color: #a0c4e0 !important; }
section[data-testid="stSidebar"] hr { border-color: #1a2e44 !important; }

/* Download button — light bg → dark text  */
div[data-testid="stDownloadButton"] > button {
    background: #1a3a5c !important;
    color: #e8f4ff !important;
    border: 1px solid #2a5080 !important;
    border-radius: 10px !important;
}

/* ── Primary export button (Markdown = default format) ── */
.st-key-dl_primary div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 0 16px rgba(6,182,212,0.3) !important;
}
.st-key-dl_primary div[data-testid="stDownloadButton"] > button:hover {
    box-shadow: 0 0 24px rgba(6,182,212,0.5) !important;
}

/* ── Sidebar nav link (Internal Observability) — plain link, not the big CTA ── */
.st-key-nav_obs div[data-testid="stButton"] > button {
    background: #0e1a2b !important;
    border: 1px solid #1a2e44 !important;
    color: #67e8f9 !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    letter-spacing: normal !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.st-key-nav_obs div[data-testid="stButton"] > button:hover {
    background: #16283f !important;
    border-color: #06b6d4 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Booking search panel (compact) ── */
.st-key-booking_panel {
    background: #0d1117;
    border: 1px solid #16222c;
    border-radius: 12px;
    padding: 1rem 1.2rem 1.1rem;
    margin-bottom: 1.1rem;
}
.st-key-booking_panel div[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
.st-key-booking_panel .stSlider,
.st-key-booking_panel .stSelectbox,
.st-key-booking_panel .stNumberInput,
.st-key-booking_panel .stDateInput {
    margin-bottom: -0.4rem;
}
.st-key-swap_cities button {
    border-radius: 50% !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    padding: 0 !important;
    background: #111b2b !important;
    border: 1px solid #1e3050 !important;
    color: #67e8f9 !important;
    box-shadow: none !important;
}
.st-key-swap_cities button:hover {
    background: #1a2e47 !important;
    border-color: #06b6d4 !important;
    transform: rotate(180deg) !important;
    transition: transform 0.4s ease !important;
}
.st-key-booking_panel div[data-baseweb="select"] > div,
.st-key-booking_panel input[type="number"] {
    background: #0a1520 !important;
    border-color: #1e2e44 !important;
    border-radius: 10px !important;
}
.st-key-booking_panel [data-testid="stSliderThumbValue"],
.st-key-booking_panel [data-testid="stThumbValue"] {
    color: #67e8f9 !important;
}
div[data-testid="stSegmentedControl"] label {
    border-radius: 20px !important;
}

/* ── Quick-suggestion chips (distinct from primary CTA) ── */
.st-key-quick_chips div[data-testid="stButton"] > button {
    background: #111b2b !important;
    border: 1px solid #1e3050 !important;
    color: #cfe4f7 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: normal !important;
    padding: 0.5rem 0.9rem !important;
    border-radius: 20px !important;
    box-shadow: none !important;
}
.st-key-quick_chips div[data-testid="stButton"] > button:hover {
    background: #1a2e47 !important;
    border-color: #06b6d4 !important;
    color: #fff !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Pipeline stepper ── */
.stepper-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0.75rem 0 1rem;
}
.step-chip {
    flex: 1;
    min-width: 130px;
    text-align: center;
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid #1e2e44;
    background: #0e1623;
    color: #5a7a96;
    transition: all 0.3s ease;
}
.step-chip.step-active {
    border-color: #06b6d4;
    color: #67e8f9;
    background: #0e1a2e;
    box-shadow: 0 0 12px rgba(6,182,212,0.3);
}
.step-chip.step-done {
    border-color: #2a5c3a;
    color: #7adf9c;
    background: #0e1f16;
}

/* ── Tabs ── */
div[data-testid="stTabs"] { margin-top: 0.5rem; }
button[data-baseweb="tab"] {
    color: #7aa8cc !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #67e8f9 !important; }
div[data-baseweb="tab-highlight"] { background-color: #06b6d4 !important; }
div[data-baseweb="tab-border"] { background-color: #1e2e44 !important; }
div[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p { color: #cce0f5 !important; }

/* ── Final-plan table formatting ── */
.final-card table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.88rem;
}
.final-card th {
    background: #12233c;
    color: #67e8f9 !important;
    padding: 0.5rem 0.75rem;
    text-align: left;
    border-bottom: 2px solid #1e3a5c;
}
.final-card td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #16283f;
    color: #cce0f5 !important;
}
.final-card tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
.final-card h1, .final-card h2, .final-card h3 {
    color: #e8f4ff !important;
    margin-top: 1.2rem;
}

/* ── Compact traveller row ── */
.trav-label {
    padding-top: 0.55rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #67e8f9;
    white-space: nowrap;
}

/* ── Favourite destination cards ── */
.dest-card {
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    height: 90px;
    cursor: pointer;
}
.dest-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.55);
    transition: transform 0.45s ease, filter 0.3s ease;
}
.dest-card:hover img {
    transform: scale(1.1);
    filter: brightness(0.7);
}
.dest-card-label {
    position: absolute;
    bottom: 8px; left: 0; right: 0;
    text-align: center;
    color: #fff;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Section header meta (e.g. active model, same line) ── */
.sec-head { justify-content: space-between; }
.sec-head-meta {
    font-size: 0.78rem;
    font-weight: 400;
    color: #a89ac4;
}
.sec-head-meta code { color: #67e8f9; background: transparent; }

/* ── Live "thinking" status line ── */
.thinking-line {
    font-size: 0.9rem;
    font-style: italic;
    color: #67e8f9;
    margin: 0.2rem 0 0.9rem;
}
.thinking-dots span {
    display: inline-block;
    font-weight: 700;
    animation: dotPulse 1.4s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.2; }
    40% { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(6,182,212,0.3); }
    50% { box-shadow: 0 0 24px rgba(6,182,212,0.7); }
}
.top-panel, .st-key-booking_panel, .final-card, .metric-box {
    animation: fadeInUp 0.5s ease both;
}
.dest-card {
    animation: fadeInUp 0.5s ease both;
}
.step-chip.step-active {
    animation: pulseGlow 1.6s ease-in-out infinite;
}

/* ── Responsive ── */
@media (max-width: 900px) {
    .top-panel {
        white-space: normal; height: auto;
        margin: 0 -1rem 1.1rem -1rem; padding: 0.85rem 1rem;
    }
    .step-chip { min-width: 45%; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
}
@media (max-width: 600px) {
    .metric-row { flex-direction: column; }
    .step-chip { min-width: 100%; }
}
</style>
""", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "sapan_mohanty"
if "view" not in st.session_state:
    st.session_state.view = "home"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    thread_id = st.text_input("👤 User ID", key="thread_id",
                               help="Your session ID — keeps travel history across queries")

    model_label = st.selectbox(
        "🧠 LLM Model (OpenRouter)", list(MODEL_OPTIONS.keys()),
        help="All routed through OpenRouter — switch here if a model is rate-limited or slow",
    )
    selected_model = MODEL_OPTIONS[model_label]

    st.markdown("<div class='sidebar-title'>Powered by</div>", unsafe_allow_html=True)
    for tech in ["🔗 LangGraph", "📊 LangSmith", "🐘 PostgreSQL", "🔍 Tavily Search",
                 "✈️ AviationStack", "🗺️ OpenTripMap"]:
        st.markdown(f"<div class='sidebar-chip'>{tech}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Agent Pipeline</div>", unsafe_allow_html=True)
    for step in ["① Flight Agent", "② Hotel Agent", "③ Weather Agent",
                 "④ Sightseeing Agent", "⑤ Itinerary Agent"]:
        st.markdown(f"<div class='sidebar-chip'>{step}</div>", unsafe_allow_html=True)

    render_observability_panel()

if st.session_state.view == "observability":
    render_observability_page()
    st.stop()

# ── Compact top panel ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='top-panel'>✈️ <b>AI Travel Booking System</b>"
    "<span class='top-panel-sep'>—</span>"
    "<span>Four specialized agents work together — searching flights, hotels, "
    "building an itinerary, and delivering your perfect trip plan.</span></div>",
    unsafe_allow_html=True,
)

if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "from_city" not in st.session_state:
    st.session_state.from_city = "Delhi, India"
if "to_city" not in st.session_state:
    st.session_state.to_city = "Tokyo, Japan"

def _swap_cities():
    st.session_state.from_city, st.session_state.to_city = (
        st.session_state.to_city, st.session_state.from_city,
    )


# ── Booking search panel ─────────────────────────────────────────────────────
with st.container(key="booking_panel"):
    route_from, route_swap, route_to = st.columns([5, 1, 5])
    with route_from:
        from_city = st.selectbox("From", CITIES, key="from_city")
    with route_swap:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        st.button("🔄", key="swap_cities", help="Swap origin and destination",
                  on_click=_swap_cities)
    with route_to:
        to_city = st.selectbox("To", CITIES, key="to_city")

    date_col1, date_col2, budget_col = st.columns([1.2, 1.2, 2.2])
    with date_col1:
        start_date = st.date_input("📅 Start Date", value=date.today() + timedelta(days=30),
                                    min_value=date.today())
    with date_col2:
        end_date = st.date_input("📅 End Date", value=date.today() + timedelta(days=37),
                                  min_value=date.today())
    with budget_col:
        budget = st.slider("💰 Budget (₹)", min_value=20_000, max_value=1_000_000,
                            value=200_000, step=5_000,
                            help="Maximum total budget for the trip")

    tc = st.columns([1.5, 1, 1.6, 1, 1.4, 1, 1.9, 1])
    with tc[0]:
        st.markdown("<div class='trav-label'>👤 Adults</div>", unsafe_allow_html=True)
    with tc[1]:
        adults = st.number_input("Adults", min_value=1, max_value=10, value=2, step=1,
                                  label_visibility="collapsed")
    with tc[2]:
        st.markdown("<div class='trav-label'>🧒 Children</div>", unsafe_allow_html=True)
    with tc[3]:
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1,
                                    help="Ages 2–11", label_visibility="collapsed")
    with tc[4]:
        st.markdown("<div class='trav-label'>👶 Infants</div>", unsafe_allow_html=True)
    with tc[5]:
        infants = st.number_input("Infants", min_value=0, max_value=5, value=0, step=1,
                                   help="Under 2", label_visibility="collapsed")
    with tc[6]:
        st.markdown("<div class='trav-label'>👴 Seniors</div>", unsafe_allow_html=True)
    with tc[7]:
        seniors = st.number_input("Senior Citizens", min_value=0, max_value=10, value=0, step=1,
                                   help="Ages 60+", label_visibility="collapsed")

    travel_type = st.segmented_control("✨ Type of Travel Plan", TRAVEL_TYPES, default="Leisure")

    search = st.button("🔍 Search Trips", use_container_width=True)

    if search:
        travellers = []
        if adults:
            travellers.append(f"{adults} adult{'s' if adults != 1 else ''}")
        if children:
            travellers.append(f"{children} child{'ren' if children != 1 else ''}")
        if infants:
            travellers.append(f"{infants} infant{'s' if infants != 1 else ''}")
        if seniors:
            travellers.append(f"{seniors} senior citizen{'s' if seniors != 1 else ''}")
        traveller_str = ", ".join(travellers) if travellers else "1 adult"

        st.session_state.user_query = (
            f"Plan a {(travel_type or 'Leisure').lower()} trip from {from_city} to {to_city} "
            f"for {traveller_str}, travelling from {start_date.strftime('%d %b %Y')} to "
            f"{end_date.strftime('%d %b %Y')}, with a total budget of up to ₹{budget:,}."
        )

# ── Favourite destinations (below the booking engine) ───────────────────────────
st.markdown("<div class='input-label' style='margin-top:1.2rem;'>⭐ Favourite Destinations</div>",
            unsafe_allow_html=True)
DESTINATIONS = [
    ("🇯🇵 Tokyo",     "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=300&q=70"),
    ("🇫🇷 Paris",     "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=300&q=70"),
    ("🇹🇭 Bangkok",   "https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=300&q=70"),
    ("🇮🇹 Rome",      "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=300&q=70"),
    ("🇦🇪 Dubai",     "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=300&q=70"),
]

cols = st.columns(5)
for col, (name, img_url) in zip(cols, DESTINATIONS):
    with col:
        st.markdown(f"""
        <div class="dest-card">
            <img src="{img_url}" />
            <div class="dest-card-label">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='input-label'>🗺️ Or describe your trip in your own words</div>", unsafe_allow_html=True)

QUICK = ["7-day Japan under ₹2L", "Paris trip for 5 days", "Dubai weekend trip", "Bali backpacking 10 days"]
with st.container(key="quick_chips"):
    qcols = st.columns(len(QUICK))
    for qc, label in zip(qcols, QUICK):
        with qc:
            if st.button(label, key=f"q_{label}", use_container_width=True):
                st.session_state.user_query = label

user_query = st.text_area(
    "",
    key="user_query",
    placeholder="e.g. Plan a complete 7-day Japan trip including flights, hotels and sightseeing under ₹2 lakhs",
    height=100,
    label_visibility="collapsed",
)

generate = st.button("🚀  Generate My Travel Plan", use_container_width=True)

# ── Agent pipeline ────────────────────────────────────────────────────────────
AGENT_META = {
    "flight_agent":      ("✈️", "Flight Agent"),
    "hotel_agent":       ("🏨", "Hotel Agent"),
    "weather_agent":     ("🌤️", "Weather Agent"),
    "sightseeing_agent": ("📍", "Sightseeing Agent"),
    "itinerary_agent":   ("🗓️", "Itinerary Agent"),
    "final_agent":       ("🧠", "Final Agent"),
}

def _to_text(value) -> str:
    """MCP tool results can come back as a string or as a list of content
    blocks (e.g. [{'type': 'text', 'text': '...'}]) — normalize to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(value) if value else ""


STEPS = [
    ("flight_agent",      "✈️", "Flights"),
    ("hotel_agent",       "🏨", "Hotels"),
    ("weather_agent",     "🌤️", "Weather"),
    ("sightseeing_agent", "📍", "Sightseeing"),
    ("itinerary_agent",   "🗓️", "Itinerary"),
]


def render_stepper(placeholder, done_steps, active_idx):
    chips = []
    for i, (key, icon, label) in enumerate(STEPS):
        if key in done_steps:
            cls, mark = "step-done", "✓ "
        elif i == active_idx:
            cls, mark = "step-active", ""
        else:
            cls, mark = "step-pending", ""
        chips.append(f"<div class='step-chip {cls}'>{mark}{icon} {label}</div>")
    placeholder.markdown(f"<div class='stepper-row'>{''.join(chips)}</div>", unsafe_allow_html=True)


THINKING_MESSAGES = {
    "flight_agent": "🔎 Searching flights and comparing airlines",
    "hotel_agent": "🏨 Finding hotels that fit your budget",
    "weather_agent": "🌤️ Checking the forecast for your dates",
    "sightseeing_agent": "📍 Finding top attractions at your destination",
    "itinerary_agent": "🧠 Thinking through your day-by-day itinerary",
}
DOTS = "<span class='thinking-dots'><span>.</span><span>.</span><span>.</span></span>"


def render_thinking(placeholder, node_key):
    msg = THINKING_MESSAGES.get(node_key, "Working")
    placeholder.markdown(f"<div class='thinking-line'>{msg}{DOTS}</div>", unsafe_allow_html=True)


if generate:
    if not user_query.strip():
        st.warning("Please describe your trip first.")
    else:
        main.set_model(selected_model)
        mcp_client.set_model(selected_model)

        config = {"configurable": {"thread_id": thread_id}}
        collected = {"flight_results": "", "hotel_results": "", "weather_results": "",
                     "sightseeing_results": "",
                     "itinerary": "", "final_response": "", "llm_calls": 0, "total_tokens": 0}

        st.markdown("---")
        st.markdown(
            f"<div class='sec-head'><span>🤖 Agent Pipeline</span>"
            f"<span class='sec-head-meta'>🧠 Running on <code>{selected_model}</code></span></div>",
            unsafe_allow_html=True,
        )

        stepper_ph = st.empty()
        thinking_ph = st.empty()
        progress_ph = st.progress(0, text="Starting agent pipeline…")
        done_steps = set()
        render_stepper(stepper_ph, done_steps, active_idx=0)
        render_thinking(thinking_ph, STEPS[0][0])

        error_msg = None
        try:
            for chunk in app.stream(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "user_query": user_query,
                    "flight_results": "",
                    "hotel_results": "",
                    "itinerary": "",
                    "llm_calls": 0,
                    "total_tokens": 0,
                    "sightseeing_results": "",
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    if node_name == "flight_agent":
                        collected["flight_results"] = _to_text(state_update.get("flight_results", ""))
                    elif node_name == "hotel_agent":
                        collected["hotel_results"] = _to_text(state_update.get("hotel_results", ""))
                    elif node_name == "weather_agent":
                        collected["weather_results"] = _to_text(state_update.get("weather_results", ""))
                    elif node_name == "sightseeing_agent":
                        collected["sightseeing_results"] = _to_text(state_update.get("sightseeing_results", ""))
                    elif node_name == "itinerary_agent":
                        text = _to_text(state_update.get("itinerary", ""))
                        collected["itinerary"] = text
                        collected["final_response"] = collected["final_response"] or text

                    collected["llm_calls"] = state_update.get("llm_calls", collected["llm_calls"])
                    collected["total_tokens"] = state_update.get("total_tokens", collected["total_tokens"])

                    done_steps.add(node_name)
                    active_idx = min(len(STEPS) - 1, len(done_steps))
                    render_stepper(stepper_ph, done_steps, active_idx)
                    icon, label = AGENT_META.get(node_name, ("🔧", node_name))
                    pct = int(len(done_steps) / len(STEPS) * 100)
                    progress_ph.progress(pct, text=f"{icon} {label} complete")

                    if len(done_steps) < len(STEPS):
                        render_thinking(thinking_ph, STEPS[active_idx][0])
                    else:
                        thinking_ph.markdown(
                            "<div class='thinking-line'>✨ Merging everything into your final plan"
                            f"{DOTS}</div>", unsafe_allow_html=True,
                        )
        except Exception as e:
            error_msg = str(e)

        thinking_ph.empty()
        if error_msg:
            progress_ph.empty()
            st.error(f"⚠️ The agent pipeline hit an error and couldn't finish: {error_msg}")
        else:
            progress_ph.progress(100, text="✅ Trip plan ready")

            try:
                main.log_search(
                    user_id=thread_id, query=user_query, model=selected_model,
                    tokens_used=collected["total_tokens"], response=collected["final_response"],
                    start_date=start_date, end_date=end_date,
                )
            except Exception:
                pass  # history logging is best-effort — never block a successful plan on it

            # Metrics
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="metric-val">5</div><div class="metric-lbl">Agents Run</div></div>
                <div class="metric-box"><div class="metric-val">{collected['llm_calls']}</div><div class="metric-lbl">LLM Calls</div></div>
                <div class="metric-box"><div class="metric-val">{collected['total_tokens']:,}</div><div class="metric-lbl">Tokens Used</div></div>
                <div class="metric-box"><div class="metric-val">✅</div><div class="metric-lbl">Status</div></div>
            </div>
            """, unsafe_allow_html=True)

            tab_plan, tab_flights, tab_hotels, tab_weather, tab_sights = st.tabs(
                ["🧠 Trip Plan", "✈️ Flights", "🏨 Hotels", "🌤️ Weather", "📍 Sightseeing"]
            )

            with tab_plan:
                if collected["final_response"]:
                    st.markdown(f"<div class='final-card'>{collected['final_response']}</div>",
                                unsafe_allow_html=True)
                else:
                    st.info("No itinerary generated.")

            with tab_flights:
                st.markdown(collected["flight_results"] or "_No flight data returned._")

            with tab_hotels:
                st.markdown(collected["hotel_results"] or "_No hotel data returned._")

            with tab_weather:
                st.markdown(collected["weather_results"] or "_No weather data returned._")

            with tab_sights:
                st.markdown(collected["sightseeing_results"] or "_No sightseeing data returned._")

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"travel_plan_{timestamp}"
            save_dir = os.path.join(os.path.dirname(__file__), "travel_plans")
            os.makedirs(save_dir, exist_ok=True)

            SECTIONS = [
                ("Flight Information", "✈️", collected["flight_results"]),
                ("Hotel Information", "🏨", collected["hotel_results"]),
                ("Weather Information", "🌤️", collected["weather_results"]),
                ("Sightseeing / Attractions", "📍", collected["sightseeing_results"]),
                ("Itinerary", "🗓️", collected["itinerary"]),
                ("Final Travel Plan", "🧠", collected["final_response"]),
            ]

            def build_markdown(emoji_headers: bool) -> str:
                lines = [
                    "# Travel Plan",
                    f"**Query:** {user_query}  ",
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
                    f"**User ID:** {thread_id}",
                    "", "---", "",
                ]
                for title, icon, content in SECTIONS:
                    heading = f"## {icon} {title}" if emoji_headers else f"## {title}"
                    lines += [heading, content or "N/A", "", "---", ""]
                lines.append(f"*LLM Calls: {collected['llm_calls']}*")
                return "\n".join(lines)

            md_content = build_markdown(emoji_headers=True)
            export_content = build_markdown(emoji_headers=False)  # plain headers — safe for PDF/DOCX fonts

            with open(os.path.join(save_dir, f"{base_name}.md"), "w", encoding="utf-8") as f:
                f.write(md_content)

            st.markdown("<div class='sec-head'><span>📁 Export Plan</span></div>", unsafe_allow_html=True)

            dl_md, dl_pdf, dl_docx = st.columns(3)
            with dl_md:
                with st.container(key="dl_primary"):
                    st.download_button("⬇️ Markdown (.md)", data=md_content,
                                        file_name=f"{base_name}.md", mime="text/markdown",
                                        use_container_width=True)
            with dl_pdf:
                st.download_button("⬇️ PDF (.pdf)", data=render_pdf(export_content),
                                    file_name=f"{base_name}.pdf", mime="application/pdf",
                                    use_container_width=True)
            with dl_docx:
                st.download_button(
                    "⬇️ Word (.docx)", data=render_docx(export_content),
                    file_name=f"{base_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

            st.markdown(f"<div class='save-bar'>📁 Auto-saved → <code>travel_plans/{base_name}.md</code></div>",
                        unsafe_allow_html=True)

            # Share — this app runs locally with no public URL, so we share a text
            # excerpt (not the file itself) via each platform's share-intent link.
            st.markdown("<div class='sec-head'><span>📤 Share</span></div>", unsafe_allow_html=True)

            excerpt = (collected["final_response"] or "")[:500].rstrip()
            share_text = (
                f"✈️ My AI-generated trip plan for: {user_query}\n\n"
                f"{excerpt}...\n\n(Generated with AI Travel Planner)"
            )
            wa_link = "https://wa.me/?text=" + urllib.parse.quote(share_text)
            tw_link = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(share_text[:260])

            sh_wa, sh_tw = st.columns(2)
            with sh_wa:
                st.link_button("📱 Share on WhatsApp", wa_link, use_container_width=True)
            with sh_tw:
                st.link_button("🐦 Share on X / Twitter", tw_link, use_container_width=True)

            with st.expander("📋 Copy plan as text"):
                st.code(share_text, language=None)
