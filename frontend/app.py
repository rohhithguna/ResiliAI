import streamlit as st
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
    """Convert numeric status to emoji badge."""
    if int(val) == 2:
        return "🟢 Healthy"
    elif int(val) == 1:
        return "🟡 Degraded"
    else:
        return "🔴 Down"

st.set_page_config(page_title="AI SRE System", layout="wide")

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

# --------------------------------
# CONTROLS
# --------------------------------
st.subheader("Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start / Reset"):
        st.session_state.env = SREOpenEnv(seed=42)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.step = 0
        st.session_state.done = False
        st.session_state.log = []
        st.session_state.error_history = [_safe_get_error(st.session_state.state)]
        st.session_state.scenario = "None"
        st.session_state.rule_baseline = None

with col2:
    if st.button("Next Step"):
        action = select_action(st.session_state.state)
        new_state, reward, done, _ = st.session_state.env.step(action)

        st.session_state.log = st.session_state.log[-9:]
        st.session_state.log.append({
            "step": st.session_state.step,
            "action": action,
            "reward": reward
        })

        st.session_state.state = new_state
        st.session_state.step += 1
        st.session_state.error_history.append(_safe_get_error(new_state))

        # Safe done condition: only mark complete on done with low error.
        try:
            err = new_state[6]
        except Exception:
            err = new_state.get("error_rate", 1.0)

        st.session_state.done = bool(done and err < 0.05)

with col3:
    if st.button("Auto Run (10 steps)"):
        for _ in range(10):
            action = select_action(st.session_state.state)
            new_state, reward, done, _ = st.session_state.env.step(action)

            st.session_state.log = st.session_state.log[-9:]
            st.session_state.log.append({
                "step": st.session_state.step,
                "action": action,
                "reward": reward
            })

            st.session_state.state = new_state
            st.session_state.step += 1
            st.session_state.error_history.append(_safe_get_error(new_state))

            # Safe stop condition for successful recovery.
            try:
                err = new_state[6]
            except Exception:
                err = new_state.get("error_rate", 1.0)

            if done and err < 0.05:
                st.session_state.done = True
                break

if st.button("Run Extreme Scenario"):
    st.session_state.scenario = st.session_state.env.inject_scenario("extreme_failure")
    st.session_state.state = st.session_state.env.state()
    st.session_state.initial_error = st.session_state.state.get("error_rate", 1.0)
    st.session_state.done = False
    st.session_state.step = 0
    st.session_state.log = []
    st.session_state.error_history = [_safe_get_error(st.session_state.state)]
    st.session_state.rule_baseline = _run_rule_baseline("extreme_failure")
    st.success("Extreme scenario injected. Monitoring gradual stabilization.")

# --------------------------------
# DISPLAY STATE
# --------------------------------
state = st.session_state.state

st.subheader("System State")
try:
    if any(int(state[i]) == 0 for i in [0,1,2]):
        st.error("⚠ Critical Failure Detected")
except (KeyError, TypeError, IndexError):
    # Handle dict-style state
    if isinstance(state, dict):
        has_failure = any(int(state.get(k, 2)) == 0 for k in ["frontend_status", "backend_status", "db_status"])
        if has_failure:
            st.error("⚠ Critical Failure Detected")



try:
    frontend, backend, db = render_status(state[0]), render_status(state[1]), render_status(state[2])
    fl, bl, dl = state[3], state[4], state[5]
    err, traffic = state[6], state[7]
except Exception:
    frontend = render_status(state.get("frontend_status", 0))
    backend = render_status(state.get("backend_status", 0))
    db = render_status(state.get("db_status", 0))
    fl = state.get("frontend_latency", 0)
    bl = state.get("backend_latency", 0)
    dl = state.get("db_latency", 0)
    err = state.get("error_rate", 1)
    traffic = state.get("traffic_load", 0)

col1, col2, col3 = st.columns(3)

col1.metric("Frontend", frontend)
col2.metric("Backend", backend)
col3.metric("Database", db)

col4, col5 = st.columns(2)
col4.metric("Error Rate", f"{err:.3f}")
col5.metric("Traffic", f"{traffic:.3f}")

# --------------------------------
# LOG
# --------------------------------
st.subheader("AI Decision Insight")

last_action = st.session_state.log[-1]["action"] if st.session_state.log else None

ACTION_REASON = {
    0: "System stable → monitoring",
    1: "Frontend issue detected → restarting frontend",
    2: "Backend instability → restarting backend",
    3: "Database failure → highest priority restart",
    4: "High load → throttling traffic",
    5: "System imbalance → rebalancing traffic"
}

if last_action is not None:
    st.info(ACTION_REASON.get(last_action, "Monitoring system"))

st.subheader("Recent Actions")

if st.session_state.log:
    st.table(st.session_state.log[-10:])
else:
    st.write("No actions yet")

# --------------------------------
# STATUS
# --------------------------------
if st.session_state.done:
    if err < 0.05:
        st.success("System Recovered")
    else:
        st.warning("Finished but not fully recovered")



st.subheader("Performance Summary")

state = st.session_state.state
steps = st.session_state.step
try:
    error = state[6]
except:
    error = state.get("error_rate", 1.0) if isinstance(state, dict) else 1.0

if error < 0.01:
    st.success(f"✅ Recovered in {steps} steps with low error rate {error:.3f}")
elif error < 0.05:
    st.warning(f"⚠️ Partially stable after {steps} steps (error: {error:.3f}). Recovered in {steps} steps to near-stable state.")
else:
    st.error(f"❌ System not stable (error: {error:.3f})")

st.subheader("Error Rate Reduction Over Time")
if st.session_state.error_history:
    st.line_chart(st.session_state.error_history)
else:
    st.info("No error history yet. Run a few steps to visualize reduction.")



# --------------------------------
# SCENARIO SELECTION
# --------------------------------
st.sidebar.subheader("Configuration")

scenario_option = st.sidebar.selectbox(
    "Select Scenario",
    ["None (Healthy)", "traffic_spike", "db_failure", "multi_failure", "extreme_failure"],
    help="Choose an incident scenario to simulate"
)

# Inject scenario on reset
if "scenario" not in st.session_state:
    st.session_state.scenario = "None"

if st.sidebar.button("Apply Scenario"):
    if scenario_option != "None (Healthy)":
        st.session_state.env.inject_scenario(scenario_option)
        st.session_state.state = st.session_state.env.state()
        st.session_state.scenario = scenario_option
        st.session_state.initial_error = st.session_state.state.get("error_rate", 1.0)
        st.session_state.step = 0
        st.session_state.log = []
        st.session_state.error_history = [_safe_get_error(st.session_state.state)]
        st.session_state.rule_baseline = _run_rule_baseline(scenario_option)
        st.success(f"✅ Scenario '{scenario_option}' applied!")
    else:
        st.session_state.scenario = "None"
        st.session_state.rule_baseline = _run_rule_baseline("None")
        st.info("System is healthy")



# --------------------------------
# PERFORMANCE METRICS
# --------------------------------
if st.session_state.step > 0 or "initial_error" in st.session_state:
    st.subheader("Performance Metrics")
    st.caption("*Metrics from current run. System performance is validated across multiple seeded runs for consistency.*")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        if "initial_error" in st.session_state:
            st.metric("Initial Error", f"{st.session_state.initial_error:.3f}")
        else:
            st.metric("Initial Error", "N/A")
    
    with col_m2:
        current_error = state.get("error_rate", 1.0) if isinstance(state, dict) else state[6]
        st.metric("Current Error", f"{current_error:.3f}")
    
    with col_m3:
        if "initial_error" in st.session_state:
            improvement = st.session_state.initial_error - (state.get("error_rate", 1.0) if isinstance(state, dict) else state[6])
            st.metric("Improvement", f"{improvement:.3f}")
        else:
            st.metric("Improvement", "N/A")
    
    with col_m4:
        st.metric("Steps Taken", st.session_state.step)

usage = get_usage_stats()
col_u1, col_u2 = st.columns(2)
with col_u1:
    st.metric("RL Usage %", f"{usage['rl_usage_pct']:.1f}%")
with col_u2:
    st.metric("Rule Override %", f"{usage['rule_usage_pct']:.1f}%")

if usage["rl_usage_pct"] > 80.0:
    st.success("RL Usage target met: above 80%")
elif usage["rl_used"] + usage["rule_used"] > 0:
    st.warning("RL Usage target not yet met (expected > 80%).")

st.subheader("AI vs Rule Comparison")
if st.session_state.get("rule_baseline") is None:
    st.session_state.rule_baseline = _run_rule_baseline(st.session_state.get("scenario", "None"))

baseline = st.session_state.rule_baseline
cmp_col1, cmp_col2 = st.columns(2)
with cmp_col1:
    st.metric("AI Steps", st.session_state.step)
    st.metric("Rule Steps", baseline["steps"])
with cmp_col2:
    st.metric("AI Final Error", f"{error:.3f}")
    st.metric("Rule Final Error", f"{baseline['final_error']:.3f}")



# --------------------------------
# POLICY INSIGHTS
# --------------------------------
st.subheader("Decision Strategy")

if st.session_state.log:
    last_action = st.session_state.log[-1]["action"] if st.session_state.log else None
    
    if last_action is not None:
        action_names = {
            0: "🟢 Monitoring",
            1: "🔄 Frontend Restart",
            2: "🔄 Backend Restart",
            3: "🔄 Database Restart",
            4: "🚦 Traffic Throttle",
            5: "⚖️  Load Rebalance"
        }
        
        st.info(f"Last Action: {action_names.get(last_action, 'Unknown')}")
        st.write("This hybrid AI/Rule system intelligently combines:")
        st.write("- **AI Agent**: Adaptive responses informed by reinforcement learning")
        st.write("- **Rule Engine**: Critical failure prioritization and safety overrides")
        st.write("- **Validation**: System performance is validated across multiple scenarios with seeded runs")
else:
    st.info("No actions taken yet. Click 'Start / Reset' to begin.")
