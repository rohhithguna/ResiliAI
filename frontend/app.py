import sys
from pathlib import Path
import streamlit as st
import altair as alt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.sre_openenv import SREOpenEnv
from src.inference.inference import select_action, get_usage_stats

# ── Helpers (unchanged logic) ──────────────────────────────────────
ACTION_LABELS = {
    0: "Monitor Strategy",
    1: "Restart Frontend",
    2: "Restart Backend",
    3: "Restart Database",
    4: "Throttle Traffic",
    5: "Rebalance Load",
}
ACTION_REASONS = {
    0: "System is stable — continuously analyzing telemetry metrics",
    1: "Frontend is down, causing dropped user connections",
    2: "Backend latency elevated — restart resolves memory leak",
    3: "Database is down, causing cascading system errors",
    4: "High traffic detected — mitigating overload pressure",
    5: "Uneven distribution detected — forcing load proxy refresh",
}
ACTION_IMPACTS = {
    0: "Maintain system stability and observe trends",
    1: "Restore user-facing connectivity and reduce 5xx errors",
    2: "Reduce request failure rate and normalize latency",
    3: "Recover data layer and unblock backend queries",
    4: "Prevent queue overflow and reduce error cascade",
    5: "Equalize server load and lower response times",
}
SCENARIO_INFO = {
    "traffic_spike": ("Traffic Spike", "Simulates sudden 3× traffic surge overwhelming backend capacity"),
    "db_failure": ("Database Failure", "Database goes offline causing cascading query failures"),
    "multi_failure": ("Multi-Service Failure", "Frontend degraded + Backend down + DB degraded simultaneously"),
    "extreme_failure": ("Extreme Failure", "Worst-case scenario: all services critically impaired with maximum traffic load"),
}

# ── Monochrome status system ───────────────────────────────────────
# Only tiny dots get subtle tint — everything else is neutral gray
_DOT_OK   = "#86efac"   # very soft green, used ONLY as 6px dot
_DOT_WARN = "#d4b96a"   # muted warm, used ONLY as 6px dot
_DOT_CRIT = "#c9706e"   # muted red, used ONLY as 6px dot

def _safe_err(state):
    if isinstance(state, dict):
        return float(state.get("error_rate", 1.0))
    return float(state[6])

def _rule_policy(state):
    if isinstance(state, dict):
        f, b, db = int(state.get("frontend_status", 2)), int(state.get("backend_status", 2)), int(state.get("db_status", 2))
        err = float(state.get("error_rate", 0.0))
    else:
        f, b, db, err = int(state[0]), int(state[1]), int(state[2]), float(state[6])
    if db == 0: return 3
    if b == 0: return 2
    if f == 0: return 1
    if err > 0.3: return 4
    return 0

def _run_rule_baseline(scenario, max_steps=40):
    env = SREOpenEnv(seed=42)
    _ = env.reset()
    if scenario and scenario != "None":
        env.inject_scenario(scenario)
    state = env.state()
    steps, done = 0, False
    for _ in range(max_steps):
        action = _rule_policy(state)
        state, _, done, _ = env.step(action)
        steps += 1
        if done and _safe_err(state) < 0.05:
            break
    return {"steps": steps, "final_error": _safe_err(state)}

def _status_label(val):
    v = int(val)
    if v == 2: return "Healthy", "#6b7280"
    elif v == 1: return "Degraded", "#6b7280"
    return "Down", "#6b7280"

def _dot_color(val):
    v = int(val)
    if v == 2: return _DOT_OK
    elif v == 1: return _DOT_WARN
    return _DOT_CRIT

def _latency_label(val):
    v = float(val)
    if v < 200: return "Normal", "#9ca3af"
    if v < 800: return "Elevated", "#9ca3af"
    return "Critical", "#9ca3af"

def _issue_explanation(name, status_val, latency_val, traffic=0, queue=0):
    s = int(status_val)
    l = float(latency_val)
    if s == 2 and l < 200: return "Operating normally"
    if s == 0: return f"Offline — requires restart"
    if s == 1 and l > 800: return f"High latency ({l:.0f}ms)"
    if s == 1: return "Performance reduced"
    if l > 800: return f"Latency spike ({l:.0f}ms)"
    if l > 200: return f"Latency elevated ({l:.0f}ms)"
    return "Stable"

def _status_dot(v):
    if v < 0.05: return _DOT_OK
    if v < 0.2: return _DOT_WARN
    return _DOT_CRIT

def _status_word(v):
    if v < 0.05: return "Stable", "#6b7280"
    if v < 0.2: return "Recovering", "#6b7280"
    return "Critical", "#6b7280"

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(page_title="AI SRE · Autonomous Recovery", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global + Grid ── */
*, html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f5f7fb 0%, #e9ecf3 100%) !important;
    color: #111;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 32px 24px 32px !important; max-width: 1080px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
div[data-testid="stHorizontalBlock"] { gap: 20px; align-items: stretch; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div { height: 100%; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div > div > div { height: 100%; }

/* ── Typography ── */
h1,h2,h3,h4 { color: #111 !important; font-weight: 700 !important; letter-spacing: -0.03em; }
h2 { font-size: 18px !important; margin-bottom: 0 !important; }
p, span, div, label { color: #6b7280; }

/* ── Pill Navbar ── */
.pill-nav {
    display: flex; align-items: center; justify-content: center; gap: 2px;
    margin: 10px auto 16px auto; max-width: 640px;
    padding: 5px 8px;
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,0.04);
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.pill-nav .pn-logo {
    font-size: 10px; font-weight: 800; color: #111;
    margin-right: 12px; letter-spacing: 0.1em; text-transform: uppercase;
    opacity: 0.25;
}
.pill-nav .pn-tab {
    padding: 6px 16px; border-radius: 999px; font-size: 11.5px; font-weight: 500;
    color: #9ca3af; background: transparent; border: none;
    cursor: pointer; transition: all 0.2s ease;
}
.pill-nav .pn-tab:hover { color: #374151; background: rgba(0,0,0,0.02); }
.pill-nav .pn-tab.active {
    color: #111; font-weight: 600;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Cards ── */
.g-card {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    padding: 24px;
}
.g-card-sm {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    padding: 20px;
}
.pm-card {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    padding: 24px 16px; text-align: center; min-height: 120px;
    display: flex; flex-direction: column; justify-content: center;
}
.svc {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    padding: 18px; position: relative; overflow: hidden;
    min-height: 120px; display: flex; flex-direction: column;
}

/* ── Metric Cards ── */
.pm-card .pm-label {
    font-size: 9px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;
}
.pm-card .pm-value {
    font-size: 44px; font-weight: 600; color: #111;
    letter-spacing: -2px; line-height: 1;
}
.pm-card .pm-sub {
    font-size: 10px; color: #b0b5bf; margin-top: 8px; font-weight: 400;
}

/* ── Service Tiles ── */
.svc-bar { display: none; }
.svc .svc-name {
    font-size: 9px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;
}
.svc .svc-st { font-size: 14px; font-weight: 600; color: #111; margin-bottom: 4px; }
.svc .svc-lat { font-size: 10px; color: #9ca3af; margin-bottom: 4px; }
.svc .svc-iss {
    font-size: 10px; color: #b0b5bf; line-height: 1.4;
    overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical;
}

/* ── Control Panel ── */
.ctrl-lbl {
    font-size: 8px; font-weight: 700; color: #b0b5bf;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;
}
.ctrl-sub {
    font-size: 8.5px; color: #b0b5bf; margin-top: 1px; margin-bottom: 6px;
    line-height: 1.3; font-weight: 400; text-align: center;
}
.ctrl-card {
    background: transparent;
    padding: 8px 0;
}
.ctrl-card-title {
    font-size: 8px; font-weight: 700; color: #b0b5bf;
    text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,0,0,0.04);
}
.ctrl-divider {
    height: 1px; background: rgba(0,0,0,0.04);
    margin: 6px 0;
}
.ctrl-feedback {
    margin-top: 6px; padding-top: 6px;
    border-top: 1px solid rgba(0,0,0,0.04);
}
.ctrl-feedback .cf-label {
    font-size: 7px; font-weight: 700; color: #b0b5bf;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 3px;
}
.ctrl-feedback .cf-action {
    font-size: 10px; font-weight: 600; color: #374151;
}
.ctrl-feedback .cf-detail {
    font-size: 8.5px; color: #9ca3af; line-height: 1.4;
}

/* \u2500\u2500 Scenario Pills (styled radio) \u2500\u2500 */
[data-testid="stSelectbox"] { display: none !important; }

/* Radio → pill strip */
.stRadio > div { display: flex !important; gap: 0 !important; }
.stRadio > label { display: none !important; }
.stRadio div[role="radiogroup"] {
    display: flex !important; gap: 6px !important; flex-wrap: wrap !important;
}
.stRadio div[role="radiogroup"] label {
    display: flex !important; align-items: center !important;
    padding: 6px 14px !important; border-radius: 999px !important;
    font-size: 10px !important; font-weight: 500 !important; color: #6b7280 !important;
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(10px) !important; -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
    white-space: nowrap !important; margin: 0 !important;
    box-shadow: none !important;
}
.stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.65) !important;
    color: #374151 !important;
}
.stRadio div[role="radiogroup"] label[data-checked="true"],
.stRadio div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,0.88) !important;
    color: #111 !important; font-weight: 600 !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
/* Hide radio circle */
.stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    font-size: 10px !important; font-weight: inherit !important; color: inherit !important;
}
.stRadio div[role="radiogroup"] input[type="radio"] { display: none !important; }
.stRadio div[role="radiogroup"] label > div:first-child { display: none !important; }

/* ── AI Decision Badge ── */
.ai-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 12px;
    background: rgba(0,0,0,0.03); color: #6b7280;
    border: 1px solid rgba(0,0,0,0.05);
}

/* ── Hero Status Block ── */
.hero {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 16px; box-shadow: 0 1px 8px rgba(0,0,0,0.03);
    padding: 28px 32px; text-align: center;
    margin-bottom: 4px;
}
.hero .hero-status {
    font-size: 10px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 10px;
}
.hero .hero-number {
    font-size: 48px; font-weight: 600; color: #111;
    letter-spacing: -2px; line-height: 1; margin-bottom: 6px;
}
.hero .hero-recovery {
    font-size: 22px; font-weight: 500; color: #374151;
    letter-spacing: -0.5px; margin-bottom: 4px;
}
.hero .hero-sub {
    font-size: 10px; color: #b0b5bf; font-weight: 400;
}

/* ── Compact Service Tile ── */
.svc-mini {
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    padding: 14px 16px;
}
.svc-mini .svc-m-name {
    font-size: 9px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;
}
.svc-mini .svc-m-val {
    font-size: 13px; font-weight: 600; color: #111;
}
.svc-mini .svc-m-lat {
    font-size: 10px; color: #b0b5bf; margin-top: 2px;
}

/* ── Subtle AI Panel ── */
.ai-subtle {
    padding: 16px 0 0 0;
    border-top: 1px solid rgba(0,0,0,0.04);
    margin-top: 8px;
}
.ai-subtle .ai-s-label {
    font-size: 9px; font-weight: 700; color: #b0b5bf;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;
}
.ai-subtle .ai-s-action {
    font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 4px;
}
.ai-subtle .ai-s-detail {
    font-size: 11px; color: #9ca3af; line-height: 1.5;
}

/* ── Status Banner ── */
.s-banner {
    display: none;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%) !important;
    color: #374151 !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
    border-radius: 16px !important;
    font-weight: 600 !important; font-size: 11px !important;
    padding: 10px 0 !important;
    transition: all 0.25s cubic-bezier(0.25,0.46,0.45,0.94) !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.03) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(180deg, #ffffff 0%, #ecedf0 100%) !important;
    color: #111 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.04) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    transform: translateY(1px) !important;
}

/* ── Chart ── */
[data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] {
    background: transparent !important; border-radius: 14px; padding: 0;
    overflow: hidden;
}
iframe { background: transparent !important; }

/* ── Graph Card ── */
.graph-card {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    padding: 16px 18px 12px 18px;
}
.graph-card .graph-title {
    font-size: 11px; font-weight: 700; color: #111;
    letter-spacing: -0.01em; margin-bottom: 2px;
}
.graph-card .graph-sub {
    font-size: 10px; color: #9ca3af; font-weight: 400;
    margin-bottom: 8px;
}
.graph-empty {
    text-align: center; padding: 40px 0;
    font-size: 11px; color: #b0b5bf;
}

/* ── Comparison Table ── */
.cmp-table { width: 100%; border-collapse: separate; border-spacing: 0; }
.cmp-table th {
    font-size: 9px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 10px 16px; text-align: left;
    border-bottom: 1px solid rgba(0,0,0,0.04);
}
.cmp-table td {
    padding: 12px 16px; font-size: 13px; color: #6b7280;
    border-bottom: 1px solid rgba(0,0,0,0.03);
}
.cmp-ai { color: #111 !important; font-weight: 600; }
.cmp-rule { color: #b0b5bf !important; }
.w-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    background: rgba(0,0,0,0.03); color: #374151;
    border: 1px solid rgba(0,0,0,0.05);
    font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
}

/* ── Decision Trace Table ── */
.trace-wrap {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.04);
    border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    padding: 16px; margin-top: 4px;
}
.trace-title {
    font-size: 9px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px;
}
.trace-table { width: 100%; border-collapse: collapse; }
.trace-table th {
    font-size: 8px; font-weight: 700; color: #b0b5bf;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 6px 10px; text-align: left;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}
.trace-table td {
    padding: 6px 10px; font-size: 11px; color: #6b7280;
    border-bottom: 1px solid rgba(0,0,0,0.03);
    line-height: 1.4;
}
.trace-table td:first-child { font-weight: 600; color: #374151; width: 36px; }
.trace-table td:nth-child(2) { font-weight: 600; color: #111; }
.trace-table .trace-result-good { color: #22c55e; font-weight: 600; }
.trace-table .trace-result-mid { color: #d4b96a; font-weight: 600; }
.trace-table .trace-result-bad { color: #c9706e; font-weight: 600; }
.trace-empty {
    text-align: center; padding: 20px 0;
    font-size: 11px; color: #b0b5bf;
}

</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────
if "env" not in st.session_state:
    st.session_state.env = SREOpenEnv(seed=42)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.log = []
    st.session_state.error_history = [_safe_err(st.session_state.state)]
    st.session_state.scenario = "None"
    st.session_state.rule_baseline = None
    st.session_state.initial_error = _safe_err(st.session_state.state)
    st.session_state.page = "Dashboard"

# Guard: ensure error_history is never empty
if "error_history" not in st.session_state or not st.session_state.error_history:
    st.session_state.error_history = [_safe_err(st.session_state.state)]
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "initial_error" not in st.session_state:
    st.session_state.initial_error = _safe_err(st.session_state.state)

# ── Phase 1: Single Pill Navbar ────────────────────────────────────
pages = ["Dashboard", "AI Decisions", "Metrics", "Performance", "AI vs Rules"]
active_page = st.session_state.page

# Functional navigation — real Streamlit buttons (visually hidden, triggered by JS)
nav_container = st.container()
with nav_container:
    ncols = st.columns(len(pages))
    for i, p in enumerate(pages):
        with ncols[i]:
            if st.button(p, key=f"_nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()

# Visual pill navbar + JS wiring to trigger hidden Streamlit buttons
import streamlit.components.v1 as components

nav_tabs_json = str(pages)
components.html(f"""
<script>
const navTexts = {nav_tabs_json};
const activePage = "{active_page}";

function setupNav() {{
    const doc = window.parent.document;

    // 1) Find and hide the real Streamlit button row
    const allButtons = doc.querySelectorAll('button[kind="secondary"]');
    const navBtnMap = {{}};
    for (const btn of allButtons) {{
        const pEl = btn.querySelector('p');
        if (pEl && navTexts.includes(pEl.textContent.trim())) {{
            navBtnMap[pEl.textContent.trim()] = btn;
            // Hide the entire horizontal block containing these buttons
            let block = btn.closest('[data-testid="stHorizontalBlock"]');
            if (block) {{
                block.style.maxHeight = '0px';
                block.style.overflow = 'hidden';
                block.style.opacity = '0';
                block.style.margin = '0';
                block.style.padding = '0';
                block.style.position = 'absolute';
                block.style.pointerEvents = 'none';
            }}
        }}
    }}

    // 2) Remove any previously injected pill-nav to avoid duplicates
    const old = doc.querySelectorAll('.pill-nav-injected');
    old.forEach(el => el.remove());

    // 3) Build the visual pill navbar
    const nav = doc.createElement('div');
    nav.className = 'pill-nav pill-nav-injected';
    nav.innerHTML = '<span class="pn-logo">AI SRE</span>';

    for (const p of navTexts) {{
        const tab = doc.createElement('span');
        tab.className = 'pn-tab' + (p === activePage ? ' active' : '');
        tab.textContent = p;
        tab.style.cursor = 'pointer';
        tab.addEventListener('click', function() {{
            const btn = navBtnMap[p];
            if (btn) {{
                // Temporarily make the button clickable
                let block = btn.closest('[data-testid="stHorizontalBlock"]');
                if (block) {{
                    block.style.pointerEvents = 'auto';
                }}
                btn.click();
            }}
        }});
        nav.appendChild(tab);
    }}

    // 4) Insert the pill nav at the top of the main container
    const mainBlock = doc.querySelector('[data-testid="stVerticalBlock"]');
    if (mainBlock && !doc.querySelector('.pill-nav-injected')) {{
        mainBlock.insertBefore(nav, mainBlock.firstChild);
    }}
}}

// Run multiple times to handle Streamlit's async rendering
setupNav();
setTimeout(setupNav, 300);
setTimeout(setupNav, 800);
setTimeout(setupNav, 1500);
</script>
""", height=0)

# ── Get current state values ───────────────────────────────────────
state = st.session_state.state
s = state if isinstance(state, dict) else {}
err = float(s.get("error_rate", 1.0)) if s else _safe_err(state)
traffic = float(s.get("traffic_load", 0.0)) if s else 0.0
init_err = st.session_state.initial_error
recovery_pct = max(0, ((init_err - err) / init_err * 100)) if init_err > 0.01 else (100.0 if err < 0.05 else 0.0)

# ════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════
if st.session_state.page == "Dashboard":

    # ── HERO STATUS BLOCK ─────────────────────────────────────
    sw, swc = _status_word(err)
    sdot = _status_dot(err)
    hero_desc = "All systems nominal" if err < 0.05 else f"Recovery in progress · {st.session_state.step} steps" if err < 0.2 else f"Failure detected · {st.session_state.step} steps"
    st.markdown(f'''<div class="hero">
        <div class="hero-status"><span style="font-size:6px;color:{sdot};vertical-align:middle;margin-right:6px">●</span> {sw}</div>
        <div class="hero-number">{err*100:.1f}%</div>
        <div class="hero-recovery">{recovery_pct:.0f}% recovered</div>
        <div class="hero-sub">{hero_desc}</div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # ── SCENARIO SELECTION ────────────────────────────────────
    st.markdown('<h2>Scenario Selection</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    _SCENARIO_MAP = {"None": "None", "Traffic Spike": "traffic_spike", "DB Failure": "db_failure", "Multi Failure": "multi_failure", "Extreme": "extreme_failure"}
    _PILL_OPTIONS = list(_SCENARIO_MAP.keys())
    scenario_label = st.radio("scenario", _PILL_OPTIONS, index=0, label_visibility="collapsed", key="scenario_select", horizontal=True)
    scenario = _SCENARIO_MAP.get(scenario_label, "None")
    
    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # ── TWO-COLUMN CORE ───────────────────────────────────────
    left, right = st.columns([0.35, 0.65])

    with left:
        # ── UNIFIED CONTROL PANEL ─────────────────────────────
        st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)

        # Section: Action buttons (2x2)
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("▶ Start", use_container_width=True, key="btn_start"):
                st.session_state.env = SREOpenEnv(seed=42)
                st.session_state.state = st.session_state.env.reset()
                st.session_state.step = 0
                st.session_state.done = False
                st.session_state.log = []
                st.session_state.error_history = [_safe_err(st.session_state.state)]
                st.session_state.scenario = "None"
                st.session_state.rule_baseline = None
                st.session_state.initial_error = _safe_err(st.session_state.state)
                if scenario and scenario != "None":
                    st.session_state.env.inject_scenario(scenario)
                    st.session_state.state = st.session_state.env.state()
                    st.session_state.initial_error = _safe_err(st.session_state.state)
                    st.session_state.error_history = [st.session_state.initial_error]
                    st.session_state.rule_baseline = _run_rule_baseline(scenario)
                    st.session_state.scenario = scenario
                st.rerun()
            st.markdown('<div class="ctrl-sub">Initialize &amp; inject</div>', unsafe_allow_html=True)
        with bc2:
            if st.button("⏭ Step", use_container_width=True, key="btn_step"):
                if not st.session_state.done:
                    action = select_action(st.session_state.state)
                    new_state, reward, done, _ = st.session_state.env.step(action)
                    err_after = _safe_err(new_state)
                    st.session_state.log.append({"step": st.session_state.step, "action": action, "reward": reward, "error_after": err_after})
                    st.session_state.state = new_state
                    st.session_state.step += 1
                    st.session_state.error_history.append(err_after)
                    st.session_state.done = bool(done and err_after < 0.05)
                st.rerun()
            st.markdown('<div class="ctrl-sub">One AI action</div>', unsafe_allow_html=True)

        bc3, bc4 = st.columns(2)
        with bc3:
            if st.button("⚡ Auto", use_container_width=True, key="btn_auto"):
                for _ in range(10):
                    if st.session_state.done: break
                    action = select_action(st.session_state.state)
                    new_state, reward, done, _ = st.session_state.env.step(action)
                    err_after = _safe_err(new_state)
                    st.session_state.log.append({"step": st.session_state.step, "action": action, "reward": reward, "error_after": err_after})
                    st.session_state.state = new_state
                    st.session_state.step += 1
                    st.session_state.error_history.append(err_after)
                    if done and err_after < 0.05:
                        st.session_state.done = True
                        break
                st.rerun()
            st.markdown('<div class="ctrl-sub">Auto recovery (10 steps)</div>', unsafe_allow_html=True)
        with bc4:
            if st.button("↺ Reset", use_container_width=True, key="btn_reset"):
                st.session_state.env = SREOpenEnv(seed=42)
                st.session_state.state = st.session_state.env.reset()
                st.session_state.step = 0
                st.session_state.done = False
                st.session_state.log = []
                st.session_state.error_history = [_safe_err(st.session_state.state)]
                st.session_state.scenario = "None"
                st.session_state.rule_baseline = None
                st.session_state.initial_error = _safe_err(st.session_state.state)
                st.rerun()
            st.markdown('<div class="ctrl-sub">Restore initial state</div>', unsafe_allow_html=True)

        # Section: Embedded AI feedback
        last_action = st.session_state.log[-1]["action"] if st.session_state.log else None
        if last_action is not None:
            aname = ACTION_LABELS.get(last_action, "Unknown")
            areason = ACTION_REASONS.get(last_action, "—")
            st.markdown(f'''<div class="ctrl-feedback">
                <div class="cf-label">Last AI Action</div>
                <div class="cf-action">{aname}</div>
                <div class="cf-detail">{areason}</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="ctrl-feedback">
                <div class="cf-label">AI Engine</div>
                <div class="cf-detail">Waiting for system start</div>
            </div>''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # ── PREMIUM RECOVERY GRAPH ───────────────────────────
        hist = st.session_state.error_history
        if len(hist) > 1:
            sub_text = f"Failure reduced from {hist[0]*100:.0f}% → {hist[-1]*100:.1f}% over {len(hist)-1} steps"
            st.markdown(f'''<div class="graph-card">
                <div class="graph-title">Recovery Trajectory</div>
                <div class="graph-sub">{sub_text}</div>
            </div>''', unsafe_allow_html=True)

            df = pd.DataFrame({"Step": list(range(len(hist))), "Failure %": [v * 100 for v in hist]})

            area = alt.Chart(df).mark_area(
                interpolate='monotone',
                line={'color': '#6D5DFC', 'strokeWidth': 2.5},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='rgba(109,93,252,0.35)', offset=0),
                        alt.GradientStop(color='rgba(109,93,252,0.0)', offset=1)
                    ],
                    x1=0, x2=0, y1=0, y2=1
                ),
                opacity=1
            ).encode(
                x=alt.X('Step:Q',
                    title='Steps',
                    axis=alt.Axis(
                        labelFont='Inter', labelFontSize=10, labelColor='#9ca3af',
                        titleFont='Inter', titleFontSize=10, titleColor='#6b7280', titleFontWeight='normal',
                        grid=False, tickColor='#e5e7eb', domainColor='#e5e7eb'
                    )
                ),
                y=alt.Y('Failure %:Q',
                    title='Failure %',
                    scale=alt.Scale(domain=[0, max(hist) * 100 * 1.05]),
                    axis=alt.Axis(
                        labelFont='Inter', labelFontSize=10, labelColor='#9ca3af',
                        titleFont='Inter', titleFontSize=10, titleColor='#6b7280', titleFontWeight='normal',
                        grid=True, gridColor='rgba(0,0,0,0.04)', gridDash=[4, 4],
                        tickColor='#e5e7eb', domainColor='#e5e7eb'
                    )
                )
            ).properties(
                height=190,
                padding={'left': 0, 'right': 8, 'top': 8, 'bottom': 0}
            ).configure(
                background='transparent'
            ).configure_view(
                strokeWidth=0
            )

            st.altair_chart(area, use_container_width=True)

        elif len(hist) == 1:
            st.markdown(f'''<div class="graph-card">
                <div class="graph-title">Recovery Trajectory</div>
                <div class="graph-sub">Baseline: {hist[0]*100:.0f}% failure rate</div>
            </div>''', unsafe_allow_html=True)

            df = pd.DataFrame({"Step": [0], "Failure %": [hist[0] * 100]})
            point = alt.Chart(df).mark_circle(
                size=60, color='#6D5DFC', opacity=0.8
            ).encode(
                x=alt.X('Step:Q', title='Steps', axis=alt.Axis(labelFont='Inter', labelFontSize=10, labelColor='#9ca3af', grid=False, domainColor='#e5e7eb')),
                y=alt.Y('Failure %:Q', title='Failure %', scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(labelFont='Inter', labelFontSize=10, labelColor='#9ca3af', grid=True, gridColor='rgba(0,0,0,0.04)', gridDash=[4,4], domainColor='#e5e7eb'))
            ).properties(height=190).configure(background='transparent').configure_view(strokeWidth=0)
            st.altair_chart(point, use_container_width=True)

        else:
            st.markdown('''<div class="graph-card">
                <div class="graph-title">Recovery Trajectory</div>
                <div class="graph-empty">Select a scenario and start system</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # ── SERVICE HEALTH GRID ───────────────────────────────────
    st.markdown('<h2>Services State</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    state = st.session_state.state
    services = [
        ("Frontend", state.get("frontend_status", 2), state.get("frontend_latency", 100)),
        ("Backend", state.get("backend_status", 2), state.get("backend_latency", 100)),
        ("Database", state.get("db_status", 2), state.get("db_latency", 100)),
        ("Load Balancer", state.get("load_balancer_status", 2), state.get("request_queue", 0)),
    ]
    sc1, sc2, sc3, sc4 = st.columns(4)
    for i, (name, sv, lv) in enumerate(services):
        sl, _ = _status_label(sv)
        dc = _dot_color(sv)
        lat_text = f"{float(lv):.0f}ms" if name != "Load Balancer" else f"{float(lv):.0f} reqs"
        tile = f'''<div class="svc-mini">
            <div class="svc-m-name">{name}</div>
            <div class="svc-m-val"><span style="font-size:5px;color:{dc};margin-right:4px">●</span>{sl}</div>
            <div class="svc-m-lat">{lat_text}</div>
        </div>'''
        with [sc1, sc2, sc3, sc4][i]:
            st.markdown(tile, unsafe_allow_html=True)

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # ── AI DECISION TRACE TABLE ───────────────────────────────
    st.markdown('<h2>Decision Trace</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    if st.session_state.log:
        # Build action target mapping
        _ACTION_TARGETS = {0: "All Services", 1: "Frontend", 2: "Backend", 3: "Database", 4: "Traffic Layer", 5: "Load Balancer"}
        rows_html = ""
        for entry in st.session_state.log:
            step_n = entry["step"] + 1
            a = entry["action"]
            aname = ACTION_LABELS.get(a, "Unknown")
            target = _ACTION_TARGETS.get(a, "System")
            reason = ACTION_REASONS.get(a, "—")
            err_after = entry.get("error_after", None)
            if err_after is not None:
                if err_after < 0.05:
                    result_cls = "trace-result-good"
                    result_txt = f"{err_after*100:.1f}% ✓"
                elif err_after < 0.3:
                    result_cls = "trace-result-mid"
                    result_txt = f"{err_after*100:.1f}%"
                else:
                    result_cls = "trace-result-bad"
                    result_txt = f"{err_after*100:.1f}%"
            else:
                # Fallback for old log entries without error_after
                idx = entry["step"] + 1
                if idx < len(st.session_state.error_history):
                    e = st.session_state.error_history[idx]
                    result_cls = "trace-result-good" if e < 0.05 else "trace-result-mid" if e < 0.3 else "trace-result-bad"
                    result_txt = f"{e*100:.1f}%" + (" ✓" if e < 0.05 else "")
                else:
                    result_cls, result_txt = "", "—"
            rows_html += f'<tr><td>{step_n}</td><td>{aname}</td><td>{target}</td><td>{reason}</td><td class="{result_cls}">{result_txt}</td></tr>'

        st.markdown(f'''<div class="trace-wrap">
            <div class="trace-title">AI Decision Trace · {len(st.session_state.log)} actions</div>
            <table class="trace-table">
                <thead><tr><th>#</th><th>Action</th><th>Target</th><th>Reason</th><th>Result</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="trace-wrap">
            <div class="trace-title">AI Decision Trace</div>
            <div class="trace-empty">No actions executed yet</div>
        </div>''', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════════
# PAGE: AI DECISIONS
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == "AI Decisions":
    st.markdown("## Decision Log")
    st.markdown('<p style="color:#6b7280;margin-bottom:28px">Complete history of AI recovery actions.</p>', unsafe_allow_html=True)

    if not st.session_state.log:
        st.markdown('<div class="g-card" style="text-align:center;padding:56px"><div style="color:#6b7280;font-size:14px">No decisions yet.</div><div style="color:#9ca3af;font-size:12px;margin-top:8px">Dashboard → Select scenario → Start System → Run steps</div></div>', unsafe_allow_html=True)
    else:
        for entry in reversed(st.session_state.log[-20:]):
            a = entry["action"]
            aname = ACTION_LABELS.get(a, "Unknown")
            areason = ACTION_REASONS.get(a, "—")
            aimpact = ACTION_IMPACTS.get(a, "—")
            step_n = entry["step"]
            err_at = st.session_state.error_history[step_n + 1] if step_n + 1 < len(st.session_state.error_history) else err
            st.markdown(f'''<div class="g-card-sm" style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <span style="font-size:14px;font-weight:600;color:#111827">Step {step_n + 1} — {aname}</span>
                    <span style="font-size:11px;color:#9ca3af">{err_at*100:.1f}%</span>
                </div>
                <div style="font-size:12px;color:#6b7280;margin-bottom:5px"><span style="color:#111827;font-weight:500">Reason</span> · {areason}</div>
                <div style="font-size:12px;color:#6b7280"><span style="color:#111827;font-weight:500">Impact</span> · {aimpact}</div>
            </div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Metrics":
    st.markdown("## Recovery Metrics")
    st.markdown('<p style="color:#6b7280;margin-bottom:28px">System recovery trajectory.</p>', unsafe_allow_html=True)

    rm1, rm2, rm3 = st.columns(3)
    with rm1:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Initial Failure</div><div class="pm-value">{init_err*100:.0f}%</div></div>', unsafe_allow_html=True)
    with rm2:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Current Failure</div><div class="pm-value">{err*100:.1f}%</div></div>', unsafe_allow_html=True)
    with rm3:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Recovery</div><div class="pm-value">{recovery_pct:.0f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:14px;font-weight:600;color:#111827;margin-bottom:4px">Failure Rate Trajectory</div>', unsafe_allow_html=True)
    hist = st.session_state.error_history
    if len(hist) > 1:
        st.markdown(f'<div style="font-size:11px;color:#9ca3af;margin-bottom:12px">{hist[0]*100:.0f}% → {hist[-1]*100:.1f}%</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({"Failure %": [v * 100 for v in hist]})
        st.area_chart(df, color="#9ca3af", use_container_width=True, height=280)
    elif len(hist) == 1:
        import pandas as pd
        st.markdown(f'<div style="font-size:11px;color:#9ca3af;margin-bottom:12px">Baseline: {hist[0]*100:.0f}%</div>', unsafe_allow_html=True)
        df = pd.DataFrame({"Failure %": [hist[0] * 100]})
        st.area_chart(df, color="#9ca3af", use_container_width=True, height=280)
    else:
        st.markdown('<div style="color:#9ca3af;font-size:13px;padding:32px 0;text-align:center">Run recovery steps to see trajectory.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if len(hist) > 1 and init_err > 0.01:
        st.markdown(f'''<div class="g-card-sm" style="margin-top:16px;text-align:center">
            <span style="font-size:14px;font-weight:500;color:#6b7280">Reduced from {init_err*100:.0f}% → {err*100:.1f}% in {st.session_state.step} steps</span>
        </div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Performance":
    st.markdown("## Performance")
    st.markdown('<p style="color:#6b7280;margin-bottom:28px">Engine utilization and system metrics.</p>', unsafe_allow_html=True)

    usage = get_usage_stats()
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Recovery</div><div class="pm-value">{recovery_pct:.0f}%</div><div class="pm-sub">Error reduction</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Steps</div><div class="pm-value">{st.session_state.step}</div><div class="pm-sub">Actions taken</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="pm-card"><div class="pm-label">RL Usage</div><div class="pm-value">{usage["rl_usage_pct"]:.0f}%</div><div class="pm-sub">{usage["rl_used"]} decisions</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="pm-card"><div class="pm-label">Rule Fallback</div><div class="pm-value">{usage["rule_usage_pct"]:.0f}%</div><div class="pm-sub">{usage["rule_used"]} decisions</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    tp1, tp2 = st.columns(2)
    with tp1:
        tl = "Low" if traffic < 0.3 else "Moderate" if traffic < 0.7 else "High"
        st.markdown(f'<div class="g-card-sm"><div class="pm-label">Traffic Load</div><div style="font-size:28px;font-weight:600;color:#111827">{traffic*100:.0f}%</div><div class="pm-sub">{tl}</div></div>', unsafe_allow_html=True)
    with tp2:
        q = float(state.get("request_queue", 0)) if isinstance(state, dict) else 0
        st.markdown(f'<div class="g-card-sm"><div class="pm-label">Queue Depth</div><div style="font-size:28px;font-weight:600;color:#111827">{q:.0f}</div><div class="pm-sub">Pending requests</div></div>', unsafe_allow_html=True)

    if init_err > 0.01 and st.session_state.step > 0:
        st.markdown(f'''<div class="g-card-sm" style="margin-top:16px;text-align:center">
            <div style="font-size:14px;font-weight:500;color:#6b7280">Reduced failure from {init_err*100:.0f}% → {err*100:.1f}% in {st.session_state.step} steps</div>
            <div style="font-size:11px;color:#9ca3af;margin-top:4px">RL: {usage["rl_used"]} · Rules: {usage["rule_used"]}</div>
        </div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE: AI vs RULES
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == "AI vs Rules":
    st.markdown("## AI vs Rule-Based")
    st.markdown('<p style="color:#6b7280;margin-bottom:28px">Efficiency comparison.</p>', unsafe_allow_html=True)

    rb = st.session_state.rule_baseline
    if rb and st.session_state.step > 0:
        ai_err = err
        ai_steps = st.session_state.step
        rule_err = rb["final_error"]
        rule_steps = rb["steps"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'''<div class="g-card" style="text-align:center">
                <div class="ai-badge">AI Agent</div>
                <div style="font-size:42px;font-weight:600;color:#111827;letter-spacing:-1.5px;margin:8px 0">{ai_err*100:.1f}%</div>
                <div style="font-size:11px;color:#9ca3af">Final Failure Rate</div>
                <div style="font-size:20px;font-weight:600;color:#374151;margin-top:16px">{ai_steps} steps</div>
                <div style="font-size:11px;color:#9ca3af">To stabilize</div>
            </div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="g-card" style="text-align:center">
                <div class="ai-badge" style="background:rgba(0,0,0,0.03);color:#9ca3af;border-color:rgba(0,0,0,0.06)">Rule-Based</div>
                <div style="font-size:42px;font-weight:600;color:#9ca3af;letter-spacing:-1.5px;margin:8px 0">{rule_err*100:.1f}%</div>
                <div style="font-size:11px;color:#b5b5bf">Final Failure Rate</div>
                <div style="font-size:20px;font-weight:600;color:#b5b5bf;margin-top:16px">{rule_steps} steps</div>
                <div style="font-size:11px;color:#c5c5cf">To stabilize</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        err_winner = "AI" if ai_err <= rule_err else "Rule"
        step_winner = "AI" if ai_steps <= rule_steps else "Rule"
        rule_rec_pct = max(0, ((init_err - rule_err) / init_err * 100)) if init_err > 0 else 0
        rec_winner = "AI" if recovery_pct >= rule_rec_pct else "Rule"
        st.markdown(f'''<div class="g-card">
            <table class="cmp-table">
                <tr><th>Metric</th><th>AI Agent</th><th>Rule-Based</th><th>Winner</th></tr>
                <tr><td style="color:#6b7280">Final Failure Rate</td><td class="cmp-ai">{ai_err*100:.1f}%</td><td class="cmp-rule">{rule_err*100:.1f}%</td><td><span class="w-badge">{err_winner}</span></td></tr>
                <tr><td style="color:#6b7280">Steps to Recover</td><td class="cmp-ai">{ai_steps}</td><td class="cmp-rule">{rule_steps}</td><td><span class="w-badge">{step_winner}</span></td></tr>
                <tr><td style="color:#6b7280">Recovery %</td><td class="cmp-ai">{recovery_pct:.0f}%</td><td class="cmp-rule">{rule_rec_pct:.0f}%</td><td><span class="w-badge">{rec_winner}</span></td></tr>
            </table>
        </div>''', unsafe_allow_html=True)

        st.markdown(f'''<div class="g-card-sm" style="margin-top:16px;text-align:center">
            <span style="font-size:13px;color:#6b7280">AI: {init_err*100:.0f}% → {ai_err*100:.1f}% in {ai_steps} steps · Rules: {rule_err*100:.1f}% in {rule_steps} steps</span>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="g-card" style="text-align:center;padding:56px">
            <div style="font-size:16px;font-weight:500;color:#6b7280;margin-bottom:8px">No comparison data</div>
            <div style="font-size:12px;color:#9ca3af">Dashboard → Select scenario → Start System → Auto Recover</div>
        </div>''', unsafe_allow_html=True)

