import sys
from pathlib import Path

import streamlit as st

# Ensure imports work whether Streamlit is launched from repo root or frontend/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.sre_openenv import SREOpenEnv
from src.inference.inference import select_action, get_usage_stats


def _safe_get_error(state):
    if isinstance(state, dict):
        return float(state.get("error_rate", 1.0))
    return float(state[6])


def _rule_policy(state):
    if isinstance(state, dict):
        frontend = int(state.get("frontend_status", 2))
        backend = int(state.get("backend_status", 2))
        db = int(state.get("db_status", 2))
        err = float(state.get("error_rate", 0.0))
    else:
        frontend = int(state[0])
        backend = int(state[1])
        db = int(state[2])
        err = float(state[6])

    if db == 0:
        return 3
    if backend == 0:
        return 2
    if frontend == 0:
        return 1
    if err > 0.3:
        return 4
    return 0


def _run_rule_baseline(scenario, max_steps=40):
    env = SREOpenEnv(seed=42)
    _ = env.reset()
    if scenario and scenario != "None":
        env.inject_scenario(scenario)
    state = env.state()
    steps = 0
    done = False

    for _ in range(max_steps):
        action = _rule_policy(state)
        state, _, done, _ = env.step(action)
        steps += 1
        if done and _safe_get_error(state) < 0.05:
            break

    return {
        "steps": steps,
        "final_error": _safe_get_error(state),
    }

def render_status(val):
    """Convert numeric status to colored HTML text."""
    if int(val) == 2:
        return '<span style="color:#22c55e">● Healthy</span>'
    elif int(val) == 1:
        return '<span style="color:#facc15">● Degraded</span>'
    else:
        return '<span style="color:#ef4444">● Down</span>'

def latency_note(val):
    try:
        fval = float(val)
        if fval < 200: return "Normal"
        elif fval < 800: return "Elevated"
        else: return "Critical"
    except:
        return "N/A"

st.set_page_config(page_title="AI SRE System", layout="wide")

st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #ffffff;
}

/* Floating top bar background */
.navbar-fake {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 70px;
    background: rgba(15,17,23,0.9);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    z-index: 999;
}

/* Push page content below navbar */
.block-container {
    background: transparent;
    padding-top: 90px !important;
}

/* Headers */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

/* Text */
p, span, div {
    color: #d1d5db;
}

.card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 12px;
}

.metric-container {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff !important;
}

/* Button styling */
.stButton button {
    background: #1f2937;
    color: white;
    border-radius: 10px !important;
    padding: 6px 12px;
    font-size: 13px;
}
.stButton button:hover {
    background: #374151;
}
</style>
""", unsafe_allow_html=True)

# Floating navbar visual background layer
st.markdown('<div class="navbar-fake"></div>', unsafe_allow_html=True)

st.title("AI System for Autonomous Incident Recovery")

# --------------------------------
# INIT STATE
# --------------------------------
if "env" not in st.session_state:
    st.session_state.env = SREOpenEnv(seed=42)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.log = []
    st.session_state.error_history = [_safe_get_error(st.session_state.state)]
    st.session_state.scenario = "None"
    st.session_state.rule_baseline = None

if "error_history" not in st.session_state:
    st.session_state.error_history = []

# Evaluate current error for the System Status Banner
state = st.session_state.state
try:
    current_error_for_banner = state[6]
except Exception:
    current_error_for_banner = state.get("error_rate", 1.0) if isinstance(state, dict) else 1.0

if current_error_for_banner < 0.05:
    st.success("System Status: System Stable")
elif current_error_for_banner < 0.2:
    st.warning("System Status: System Degraded")
else:
    st.error("System Status: System Critical")

# --------------------------------
# CONTROL PANEL (TOP)
# --------------------------------
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button("Start", use_container_width=True):
        st.session_state.env = SREOpenEnv(seed=42)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.step = 0
        st.session_state.done = False
        st.session_state.log = []
        st.session_state.error_history = [_safe_get_error(st.session_state.state)]
        st.session_state.scenario = "None"
        st.session_state.rule_baseline = None
        st.session_state.initial_error = _safe_get_error(st.session_state.state)
        st.success("System initialized")

with nav2:
    if st.button("Step", use_container_width=True):
        action = select_action(st.session_state.state)
        new_state, reward, done, _ = st.session_state.env.step(action)
        st.session_state.log = st.session_state.log[-9:]
        st.session_state.log.append({"step": st.session_state.step, "action": action, "reward": reward})
        st.session_state.state = new_state
        st.session_state.step += 1
        st.session_state.error_history.append(_safe_get_error(new_state))
        try: err = new_state[6]
        except Exception: err = new_state.get("error_rate", 1.0)
        st.session_state.done = bool(done and err < 0.05)

with nav3:
    if st.button("Auto", use_container_width=True):
        for _ in range(10):
            action = select_action(st.session_state.state)
            new_state, reward, done, _ = st.session_state.env.step(action)
            st.session_state.log = st.session_state.log[-9:]
            st.session_state.log.append({"step": st.session_state.step, "action": action, "reward": reward})
            st.session_state.state = new_state
            st.session_state.step += 1
            st.session_state.error_history.append(_safe_get_error(new_state))
            try: err = new_state[6]
            except Exception: err = new_state.get("error_rate", 1.0)
            if done and err < 0.05:
                st.session_state.done = True
                break
        st.info("Running automated recovery...")

with nav4:
    if st.button("Extreme", use_container_width=True):
        st.session_state.scenario = st.session_state.env.inject_scenario("extreme_failure")
        st.session_state.state = st.session_state.env.state()
        st.session_state.initial_error = _safe_get_error(st.session_state.state)
        st.session_state.done = False
        st.session_state.step = 0
        st.session_state.log = []
        st.session_state.error_history = [_safe_get_error(st.session_state.state)]
        st.session_state.rule_baseline = _run_rule_baseline("extreme_failure")
        st.error("Extreme failure injected")

# Active State feedback
if st.session_state.log:
    last_act = st.session_state.log[-1]["action"]
    act_name = {
            0: "Monitor Strategy",
            1: "Restart Frontend",
            2: "Restart Backend",
            3: "Restart Database",
            4: "Throttle Traffic",
            5: "Rebalance Load"
    }.get(last_act, "Unknown Action")
    st.markdown(f'<div style="text-align: center; color: #a1a1aa; margin-bottom: 20px;">Last Action: {act_name}</div>', unsafe_allow_html=True)

# --------------------------------
# SYSTEM STATE & GRAPH (MIDDLE)
# --------------------------------
row2 = st.columns([1,1])
row3 = st.columns([1,1])
with row2[0]:
    st.subheader("System Health")
    state = st.session_state.state

    try:
        frontend, backend, db = render_status(state[0]), render_status(state[1]), render_status(state[2])
        fl, bl, dl = state[3], state[4], state[5]
        lb_status_raw = state[8] if len(state) > 8 else None
        lb_latency_raw = state[9] if len(state) > 9 else None
    except Exception:
        frontend = render_status(state.get("frontend_status", 0))
        backend = render_status(state.get("backend_status", 0))
        db = render_status(state.get("db_status", 0))
        fl = state.get("frontend_latency", 0)
        bl = state.get("backend_latency", 0)
        dl = state.get("db_latency", 0)
        lb_status_raw = state.get("load_balancer_status", state.get("lb_status"))
        lb_latency_raw = state.get("load_balancer_latency", state.get("lb_latency"))

    lb_status = render_status(lb_status_raw) if lb_status_raw is not None else '<span style="color:#d1d5db">⚪ Unknown</span>'
    lb_latency = float(lb_latency_raw) if lb_latency_raw is not None else 0.0

    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0; font-size: 18px;">Frontend</h3>
            <div>{frontend}</div>
            <div style="margin-top: 8px;">Latency: {float(fl):.0f} ms</div>
            <div style="font-size: 14px; opacity: 0.8;">{latency_note(fl)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with hc2:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0; font-size: 18px;">Backend</h3>
            <div>{backend}</div>
            <div style="margin-top: 8px;">Latency: {float(bl):.0f} ms</div>
            <div style="font-size: 14px; opacity: 0.8;">{latency_note(bl)}</div>
        </div>
        """, unsafe_allow_html=True)

    with hc3:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0; font-size: 18px;">Database</h3>
            <div>{db}</div>
            <div style="margin-top: 8px;">Latency: {float(dl):.0f} ms</div>
            <div style="font-size: 14px; opacity: 0.8;">{latency_note(dl)}</div>
        </div>
        """, unsafe_allow_html=True)

    with hc4:
        lb_note = latency_note(lb_latency) if lb_status_raw is not None else "N/A"
        lb_lat_str = f"Latency: {float(lb_latency):.0f} ms" if lb_status_raw is not None else "Latency: N/A"
        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0; font-size: 18px;">Load Balancer</h3>
            <div>{lb_status}</div>
            <div style="margin-top: 8px;">{lb_lat_str}</div>
            <div style="font-size: 14px; opacity: 0.8;">{lb_note}</div>
        </div>
        """, unsafe_allow_html=True)

with row2[1]:
    st.subheader("Error Reduction Over Time")
    if st.session_state.error_history:
        st.line_chart(st.session_state.error_history)
    else:
        st.info("No error history yet.")

# --------------------------------
# INSIGHTS & METRICS (BOTTOM)
# --------------------------------
with row3[0]:
    st.subheader("AI Decision Insight")
    last_action = st.session_state.log[-1]["action"] if st.session_state.log else None

    if last_action is not None:
        action_title = {
            0: "Monitor Strategy",
            1: "Restart Frontend",
            2: "Restart Backend",
            3: "Restart Database",
            4: "Throttle Traffic",
            5: "Rebalance Load"
        }.get(last_action, "Unknown Action")
        
        reason = {
            0: "System is stable, analyzing metrics",
            1: "Frontend is down, causing dropped connections",
            2: "Backend latency elevated, restarting resolves memory leak",
            3: "Database is down, causing high system error",
            4: "High traffic detected, mitigating overload",
            5: "Uneven distribution detected, forcing load proxy refresh"
        }.get(last_action, "No reason defined")

        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0; font-size: 18px;">Action: {action_title}</h3>
            <p style="margin-top: 8px; font-size: 15px;">Reason: "{reason}"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">Monitoring system...</div>', unsafe_allow_html=True)

with row3[1]:
    st.subheader("Performance Metrics")
    state = st.session_state.state
    try:
        error = state[6]
        traffic = state[7]
    except:
        error = state.get("error_rate", 1.0) if isinstance(state, dict) else 1.0
        traffic = state.get("traffic_load", 0.0) if isinstance(state, dict) else 0.0

    if error < 0.05:
        error_status = "Stable"
    elif error < 0.2:
        error_status = "Warning"
    else:
        error_status = "Critical"

    if traffic < 0.3:
        traffic_status = "Low"
    elif traffic < 0.7:
        traffic_status = "Moderate Load"
    else:
        traffic_status = "High Load"

    usage = get_usage_stats()
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 14px; opacity: 0.8;">RL Usage %</div>
            <div class="metric-container">{usage['rl_usage_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 14px; opacity: 0.8;">Steps</div>
            <div class="metric-container">{st.session_state.step}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p3:
        if "initial_error" in st.session_state:
            orig_err = st.session_state.initial_error
            imp_val = ((orig_err - error) / orig_err * 100.0) if orig_err > 0 else 0.0
            imp_str = f"{imp_val:.0f}%"
        else:
            imp_str = "N/A"
            
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 14px; opacity: 0.8;">Improvement</div>
            <div class="metric-container">{imp_str}</div>
        </div>
        """, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 14px; opacity: 0.8;">System Error</div>
            <div class="metric-container">{error:.3f}</div>
            <div style="font-size: 14px; margin-top: 4px;">Status: <span style="color:#ffffff">{error_status}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with ec2:
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 14px; opacity: 0.8;">System Traffic</div>
            <div class="metric-container">{traffic:.3f}</div>
            <div style="font-size: 14px; margin-top: 4px;">Load: <span style="color:#ffffff">{traffic_status}</span></div>
        </div>
        """, unsafe_allow_html=True)
